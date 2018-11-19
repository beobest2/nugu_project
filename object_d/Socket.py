# coding: utf-8

import socket, struct, errno
import sys

class Socket(object) :
    class SocketDisconnectException(Exception) : pass
    class SocketDataSendException(Exception) : pass
    class SocketTimeoutException(Exception) : pass

    default_bufsize = 8096

    def __init__(self, bufsize=-1) :
        object.__init__(self)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        linger = struct.pack( b"ii", 1, 0 )
        self.sock.setsockopt( socket.SOL_SOCKET, socket.SO_LINGER, linger )

        self.remain = 0
        self.tmpList = []
        self.addr = ""
        self.inbuf = b""
        self.isConnect = False

        if bufsize == 0:
            self._rbufsize = 1
        elif bufsize == -1:
            self._rbufsize = self.default_bufsize
        else:
            self._rbufsize = bufsize

    def Connect(self, ip, port) :

        if not self.isConnect:
            self.sock.connect((ip, int(port)))
            self.setSock(self.sock)

    def Bind(self, port):
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind( ( '', int(port)) )
        self.sock.listen(1000)

        self.setSock(self.sock)

    def Accept(self):
        (cSock, addr) =  self.sock.accept()
        c = Socket()
        c.setSock(cSock)
        c.addr = addr
        return c

    def setSock(self, sock):
        self.sock = sock
        self.isConnect = True

    #######################################################
    #
    # 한줄 단위로 Return
    #
    def Readline(self, modeBlock=True, timeOut=0) :
        data = ""
        local_sock  = self.sock
        local_inbuf = self.inbuf

        lf = local_inbuf.find(b"\n")
        if lf >= 0:
            data = local_inbuf[:lf+1]
            local_inbuf = local_inbuf[lf+1:]
            self.inbuf = local_inbuf
            self.sock  = local_sock
            return data

        # 블럭 모드에서도 타임아웃 사용할 수 있도록 수정.
        #if not modeBlock and timeOut > 0:
        if timeOut > 0:
            local_sock.settimeout(timeOut)
        else:
            local_sock.settimeout(None)

        while True:
            try : r = local_sock.recv(self._rbufsize)
            except socket.timeout:
                self.inbuf = local_inbuf
                self.sock  = local_sock
                raise Socket.SocketTimeoutException
            except socket.error as e:
                if e.args[0] == errno.ECONNRESET:
                    self.inbuf = local_inbuf
                    self.sock  = local_sock
                    raise Socket.SocketDisconnectException
            if not r:
                self.inbuf = local_inbuf
                self.sock  = local_sock
                # connection broken
                # disconnect
                local_sock.settimeout(None)
                raise Socket.SocketDisconnectException
        
            local_inbuf = local_inbuf + r
    
            lf = r.find(b'\n')
            if lf >= 0:
                lf = local_inbuf.find(b'\n')
                data = local_inbuf[:lf + 1]
                local_inbuf = local_inbuf[lf + 1:]
                break

            # 함수가 Non-Block Mode 로 동작하는 경우 리턴
            if not modeBlock : break

        local_sock.settimeout(None)
        self.inbuf = local_inbuf
        self.sock  = local_sock

        return data

    def Read(self, size, modeBlock=True, timeOut=0) :

        self.remain = size

        tmpData = ""
        self.tmpList = []

        if len(self.inbuf) > 0:
            if self.remain > len(self.inbuf):
                self.remain = self.remain - len(self.inbuf)
                self.tmpList.append( self.inbuf )
                self.inbuf = ""
            else :  #  <=
                tmpData = self.inbuf[:self.remain]
                self.inbuf = self.inbuf[self.remain:]
                return tmpData

        #print(dir(self.sock))

        if not modeBlock and timeOut > 0:
            self.sock.settimeout(timeOut)
        else:
            self.sock.settimeout(None)

        while 1 : 
            tmpData = b''
            #try: tmpData = self.sock.recv(min(self.remain, 1024*1024), socket.MSG_WAITALL)
            try: tmpData = self.sock.recv(min(self.remain, 2147483647))
            except socket.timeout:
                raise Socket.SocketTimeoutException
            except socket.error as e:
                if e.args[0] == errno.ECONNRESET:
                    raise Socket.SocketDisconnectException

            if tmpData == b"":
                # connection broken
                # disconnect
                self.sock.settimeout(None)
                raise Socket.SocketDisconnectException

            self.tmpList.append(tmpData)
            self.remain -= len(tmpData)

            # 정상적인 경우
            if self.remain <= 0: break

        self.remain = 0
        str = b''.join(self.tmpList)
        self.tmpList = []

        self.sock.settimeout(None)

        return str

    def ReadMessage(self) :
        line = self.Readline()
        print(line)
        line = str(line, 'utf-8')
        (code, msg) = line.split(" ", 1)

        if code[0] == "+":
            return (True, msg)
        else:
            return (False, msg)

    def SendMessage(self, cmd, timeOut=0) :

        if timeOut > 0:
            self.sock.settimeout(timeOut)
        else:
            self.sock.settimeout(None)

        while True:
            print(cmd)
            n =  self.sock.send(cmd)
            print(n)
            if n == len(cmd):
                break
            elif n <= 0:
                self.sock.settimeout(None)
                raise Socket.SocketDataSendException

            cmd = cmd[n:]

        self.sock.settimeout(None)

    def close(self):
        self.sock.close()
        self.isConnect = False
    """
    def shutdown(self, num=2):
        self.sock.shutdown(num)
    """
def server():
    s = Socket()
    s.Bind(9999)

    client_sock = s.Accept()
    print("Accept")
    client_sock.SendMessage(b"+OK Hello World!!!!\r\n")

    while True:
        msg = client_sock.Readline()
        client_sock.SendMessage(b"+OK %s" % msg )
        print("MSG:  ", msg)
        msg = msg.decode("utf-8")
        if msg.strip().upper() == "QUIT":
            break

    client_sock.close()
    s.close()


def client():
    import time
    s = Socket()
    s.Connect("localhost", 9999)
    print(s.ReadMessage())
    s.SendMessage(b"GET\r\n")
    print(s.ReadMessage())
    s.SendMessage(b"QUIT\r\n")
    print(s.ReadMessage())
    time.sleep(15)
    s.close()


if __name__ == "__main__":
    print(sys.argv)
    if sys.argv[1] == '1':
        client()
    else:
        server()
