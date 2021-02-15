import csv
import socket, numpy as np
from sklearn.linear_model import LinearRegression
import ctypes
from Agent.DDPG.ddpg_torch import Agent

class socketserver:
    def __init__(self, address='', port=9090):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.address = address
        self.port = port
        self.sock.bind((self.address, self.port))
        self.cummdata = ''

    def recvmsg(self):
        self.sock.listen(1)
        self.conn, self.addr = self.sock.accept()
        print('connected to', self.addr)
        self.cummdata = ''

        while True:
            data = self.conn.recv(10000000)
            self.cummdata += data.decode("utf-8")
            if not data:
                break
            #self.conn.send(bytes(self.calcregr(self.cummdata), "utf-8"))
            dataList = list(self.cummdata.split(" "))
            dataList.pop(len(dataList)-1)
            dataList = list(map(float, dataList))
            return dataList

    def __del__(self):
        self.sock.close()

    def calcregr(self, msg=''):
        chartdata = np.fromstring(msg, dtype=float, sep=' ')
        Y = np.array(chartdata).reshape(-1, 1)
        X = np.array(np.arange(len(chartdata))).reshape(-1, 1)

        lr = LinearRegression()
        lr.fit(X, Y)
        Y_pred = lr.predict(X)
        type(Y_pred)
        P = Y_pred.astype(str).item(-1) + ' ' + Y_pred.astype(str).item(0)
        print(P)
        return str(P)

class Env():
    def __init__(self):
        self.steps=0

    def step(self, action):
        a = np.argmax(action)
        if a == 0:
            action_send = "BUY"
        elif a == 1:
            action_send = "SELL"
        elif a == 2:
            action_send = "CLOSE"
        else:
            action_send = "HOLD"

        return action_send


ctypes.windll.shell32.IsUserAnAdmin()
serv = socketserver('127.0.0.1', 9999)
import time

print("wating")
state = serv.recvmsg()
agent = Agent(alpha=0.0001, beta=0.001, input_dims=[len(state)], tau=0.001, batch_size=64, fc1_dims=800,fc2_dims=300, n_actions=4)
env = Env()
higest_profit=100;


while True:
    f = open("save_log.txt", "r")
    validation = f.read()
    f.close()
    if validation == "Done":
        f = open("save_log.txt", "w")
        f.write("Loading")
        f.close()
        agent.load_models()
        f = open("save_log.txt", "w")
        f.write("Done")
        f.close()
    action = agent.choose_action(np.array(state))
    action_Send = env.step(action)
    print("-"*40)
    serv.conn.send(bytes(action_Send, "utf-8"))
    newState = serv.recvmsg()
    with open("data.csv", "a") as fp:
        wr = csv.writer(fp, dialect='excel')
        wr.writerow(state)
    fp.close()

    if newState[39]>0:
        done = True
    else:
        done=False
    print("Reward : {0} Action: {1} oldProfit. {2} new Profit {3} ".format(newState[38], action_Send,state[0], newState[0]))
    agent.remember(state=np.array(state), action=action, reward=newState[38], new_state=np.array(newState), done=done)
    agent.learn()
    if newState[38]>higest_profit :
        agent.save_models()
        higest_profit=newState[38]
    state=newState



