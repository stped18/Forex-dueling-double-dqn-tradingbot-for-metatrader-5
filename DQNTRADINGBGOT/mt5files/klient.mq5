//+------------------------------------------------------------------+
//|                                                       klient.mq5 |
//|                                          Steffen Vitten Pedersen |
//|                                         https://www.itcurity.com |
//+------------------------------------------------------------------+
#property copyright "Steffen Vitten Pedersen"
#property link      "https://www.itcurity.com"
#property version   "1.00"
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
#include <Indicator.mqh>
int socket;
string lastAction="HOLD";
double static reward=0;
int static steps=0;
double balance = AccountInfoDouble(ACCOUNT_BALANCE);
double done=0;
int OnInit()
  {
//---
   
//---
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
//---
   
  }
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
//---
socket=SocketCreate();
 if(socket!=INVALID_HANDLE) {
  if(SocketConnect(socket,"127.0.0.1",9999,10000)) {
   Print("Connected to "," 127.0.0.1",":",9999);
         
   double data[41];
   data[0]=getProfit();
   data[1]=NormalizeDouble(SymbolInfoDouble(_Symbol, SYMBOL_BID),_Digits);
   data[2]=NormalizeDouble(SymbolInfoDouble(_Symbol, SYMBOL_ASK),_Digits);
   data[3] = iOpen(Symbol(),1,1);
   data[4] = iClose(Symbol(),1,1);
   data[5] = iHigh(Symbol(),1,1);
   data[6] = iLow(Symbol(),1,1);
   data[7] = SMA(5,0);
   data[8] = SMA(8,0);
   data[9] = SMA(13,0);
   data[10] = SMA(50,0);
   data[11] = SMA(200,0);
   data[12] = EMA(8,0);
   data[13] = EMA(13,0);
   data[14] = EMA(21,0);
   data[15] = EMA(50,0);
   data[16] = EMA(150,0);
   data[17] = EMA5MIN(8,0);
   data[18] = EMA5MIN(13,0);
   data[19] = EMA5MIN(21,0);
   data[20] = STOCHASTIC(5,3,3);
   data[21] = RSI(7);
   data[22] = RSI(14);
   data[23] = RSI(21);
   data[24] = Volume();
   data[25] = Momentum(7);
   data[26] = Momentum(14);
   data[27] = Momentum(21);
   double BB[3];
   BollingerBrand(21,0,2, BB);
   data[28] = BB[0];
   data[29] = BB[1];
   data[30] = BB[2];
   BollingerBrand(14,0,2, BB);
   data[31] = BB[0];
   data[32] = BB[1];
   data[33] = BB[2];
   data[34] = CCI(7);
   data[35] = CCI(14);
   data[36] = CCI(21);
   data[37] = PositionsTotal();
   data[38] = reward;
   data[39] = done;
   data[40] = steps;
   reward=0;
         
   string tosend;
   for(int i=0;i<ArraySize(data);i++) tosend+=(string)data[i]+" ";       
   string received = socksend(socket, tosend) ? socketreceive(socket, 1000) : ""; 
   printf(received);
   actionHandler(received); 
   }
   
  else Print("Connection ","localhost",":",9090," error ",GetLastError());
  SocketClose(socket); }
 else Print("Socket creation error ",GetLastError()); 
   
  }
//+------------------------------------------------------------------+
bool socksend(int sock,string request) 
  {
   char req[];
   int  len=StringToCharArray(request,req)-1;
   if(len<0) return(false);
   return(SocketSend(sock,req,len)==len); 

  }
  
 string socketreceive(int sock,int timeout)
  {
   char rsp[];
   string result="";
   uint len;
   uint timeout_check=GetTickCount()+timeout;
   do
     {
      len=SocketIsReadable(sock);
      if(len)
        {
         int rsp_len;
         rsp_len=SocketRead(sock,rsp,len,timeout);
         if(rsp_len>0) 
           {
            result+=CharArrayToString(rsp,0,rsp_len); 
           }
        }
     }
   while((GetTickCount()<timeout_check) && !IsStopped());
   return result;
  }
 
void actionHandler(string action){
   steps +=1;
   updateAskBid();
   if(action=="BUY"){
      if (PositionsTotal()==0){
         steps=0;
         done=0;
         Buy(0.01);
         lastAction="BUY";
         reward=10+getProfit();
      }
      reward=-100;
      
      
   }
   if (action=="CLOSE"){
      if (PositionsTotal()>0){
      done=1;
      steps=0;
      lastAction="CLOSE";
      Close();
      reward=getProfit()*100;
      }
      reward=-100+getProfit();
      
   }
   if(action=="SELL"){
   if (PositionsTotal()==0){
         steps=0;
         Sell(0.01);
         done=0;
         lastAction="BUY";
         reward=10+getProfit();
      }
      reward=-100;
   }
   if(action=="HOLD"){
      if (PositionsTotal()>0){
         reward=getProfit()-(steps/100);
      }
   
   reward=getProfit()-(steps/100);
   }
   
   if (AccountInfoDouble(ACCOUNT_BALANCE)>balance){
      balance=AccountInfoDouble(ACCOUNT_BALANCE);
      done = 1;
      reward = reward*2;
   }

}