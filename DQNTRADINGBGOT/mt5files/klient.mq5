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
double static steps=0;
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
         
   double data[62];
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
   printf("reward "+reward);
   data[39] = done;
   data[40] = steps;
   data[41] = LarryWilliams(21);
   data[42] = LarryWilliams(14);
   data[43] = LarryWilliams(7);
   data[44] = SAR();
   data[45] = AO();
   data[46] = BearsPower(21);
   data[47] = BearsPower(14);
   data[48] = BearsPower(7);
   data[49] = BearsPower(21);
   data[50] = BearsPower(14);
   data[51] = BearsPower(7);
   data[52] = RVI(21);
   data[53] = RVI(14);
   data[54] = RVI(7);
   double Adx[2];
   ADX(21,Adx);
   data[55] = Adx[0];
   data[56] = Adx[1];
   ADX(14,Adx);
   data[57] = Adx[0];
   data[58] = Adx[1];
   ADX(7,Adx);
   data[59] = Adx[0];
   data[60] = Adx[1];
   data[61] = AccountInfoDouble(ACCOUNT_BALANCE);
   
   
   
   
         
   string tosend;
   for(int i=0;i<ArraySize(data);i++) tosend+=(string)data[i]+" ";       
   string received = socksend(socket, tosend) ? socketreceive(socket, 10000) : ""; 
   printf("resived : "+received);
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
      if (PositionsTotal()==1){
         reward=(getProfit()*10)-(steps/10000);
         Close();
         
      }
      if (PositionsTotal()==0){
         
         steps=0;
         done=1;
         Buy(0.01);
         lastAction="BUY";
         
      }else{
      reward=-100*steps;
      }
   }
   else if(action=="SELL"){
      if (PositionsTotal()==1){
         reward=(getProfit()*10)-(steps/10000);
         Close();
         
      }
      if (PositionsTotal()==0){
  
         steps=0;
         done=1;
         lastAction="SELL";
         Sell(0.01);
         
         }
       else{reward=-100*steps;}
      
   }
   else{
      done=0;
      if (PositionsTotal()>0){
         reward=getProfit()-(steps/10000);
         
      }
      else{
      reward=getProfit()-(steps);
      }
   }
   
 

}