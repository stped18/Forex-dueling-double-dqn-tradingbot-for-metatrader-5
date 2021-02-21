//+------------------------------------------------------------------+
//|                                                    Indicator.mqh |
//|                                          Steffen Vitten Pedersen |
//|                                         https://www.itcurity.com |
//+------------------------------------------------------------------+
#property copyright "Steffen Vitten Pedersen"
#property link      "https://www.itcurity.com"
#include <Trade\Trade.mqh>
#include <Trade\PositionInfo.mqh>
#include <Trade\OrderInfo.mqh>

double Ask = NormalizeDouble(SymbolInfoDouble(_Symbol, SYMBOL_ASK),_Digits);
double Bid = NormalizeDouble(SymbolInfoDouble(_Symbol, SYMBOL_BID),_Digits);

COrderInfo order;
CTrade trade;
CPositionInfo pos;




  
  
  
void BollingerBrand(int timePeriod,int shift, double duv, double& BBdata[]){
   double MiddelBrandArray[];
   double UpperBandArray[];
   double LowerBandArray[];
   
   ArraySetAsSeries(MiddelBrandArray,true);
   ArraySetAsSeries(UpperBandArray,true);
   ArraySetAsSeries(LowerBandArray,true);
   
   int BollingerBrandDefinition = iBands(_Symbol,PERIOD_CURRENT,timePeriod,shift,duv,PRICE_CLOSE);
  
   
   CopyBuffer(BollingerBrandDefinition,0,0,3,MiddelBrandArray);
   CopyBuffer(BollingerBrandDefinition,1,0,3,UpperBandArray);
   CopyBuffer(BollingerBrandDefinition,2,0,3,LowerBandArray);
   
   double MiddelBrandValue=MiddelBrandArray[0];
   double UpperBrandValue = UpperBandArray[0];
   double LowerBandValue = LowerBandArray[0];
   
   BBdata[0]=MiddelBrandArray[0];
   BBdata[1]=UpperBandArray[0];
   BBdata[2]= LowerBandArray[0];
   
 }
 
 double MACD(int fast_ema_period, int slow_ema_period, int signal_period, int applied_price ){
   double MACDArray[];
   ArraySetAsSeries(MACDArray,true);
 
   int MACDDefinition = iMACD(_Symbol,PERIOD_CURRENT,fast_ema_period,slow_ema_period, signal_period, applied_price);
   
   CopyBuffer(MACDDefinition,0,0,3,MACDArray);
   return MACDArray[0];
   
 }
 
 double STOCHASTIC(int Kperiod, int Dperiod, int slowing){
   double DataArray[];
   ArraySetAsSeries(DataArray,true);
 
   int Definition = iStochastic(NULL,PERIOD_CURRENT,Kperiod,Dperiod, slowing, MODE_SMA,STO_LOWHIGH);
   
   CopyBuffer(Definition,0,0,3,DataArray);
   return DataArray[0];
 }
 
 

  
double SMA(int ma, int ma_shift){
   double DataArray[];
   ArraySetAsSeries(DataArray,true);
 
   int Definition = iMA(NULL,PERIOD_CURRENT,ma,ma_shift, MODE_SMA,PRICE_CLOSE);
   
   CopyBuffer(Definition,0,0,3,DataArray);
   return DataArray[0];
   
 }

  double EMA(int ma_period, int ma_shift){
   double DataArray[];
   ArraySetAsSeries(DataArray,true);
 
   int Definition = iMA(NULL,PERIOD_CURRENT,ma_period,ma_shift, MODE_EMA,PRICE_CLOSE);
   
   CopyBuffer(Definition,0,0,3,DataArray);
   return DataArray[0];
   
 }
   double EMA5MIN(int ma_period, int ma_shift){
   double DataArray[];
   ArraySetAsSeries(DataArray,true);
 
   int Definition = iMA(NULL,PERIOD_M5,ma_period,ma_shift, MODE_EMA,PRICE_CLOSE);
   
   CopyBuffer(Definition,0,0,3,DataArray);
   return DataArray[0];
   
 }
 
void Buy(double lot)
  {
   Ask = NormalizeDouble(SymbolInfoDouble(_Symbol, SYMBOL_ASK),_Digits);
   
   Print("Entert Buy");
   if(!trade.Buy(lot,NULL,Ask,0,0,NULL))
     {
      Print("Buy() method failed. Return code=",trade.ResultRetcode(),
            ". Code description: ",trade.ResultRetcodeDescription());
            
     }
   else
     {
      Print("Buy() method executed successfully. Return code=",trade.ResultRetcode(),
            " (",trade.ResultRetcodeDescription(),")");
     }
  }
//+------------------------------------------------------------------+
void Sell(double lot)
  {
  Bid = NormalizeDouble(SymbolInfoDouble(_Symbol, SYMBOL_BID),_Digits);
   Print("Entert Sell");
   if(!trade.Sell(lot,NULL,Bid,0,0,NULL))
     {
      Print("Sell() method failed. Return code=",trade.ResultRetcode(),
            ". Code description: ",trade.ResultRetcodeDescription());
     }
   else
     {
      Print("Sell() method executed successfully. Return code=",trade.ResultRetcode(),
            " (",trade.ResultRetcodeDescription(),")");
     }
  }
 

 void updateAskBid()
 {
 Ask = NormalizeDouble(SymbolInfoDouble(_Symbol, SYMBOL_ASK),_Digits);
 Bid = NormalizeDouble(SymbolInfoDouble(_Symbol, SYMBOL_BID),_Digits);
 
 }
 
 void CloseBuy(double TP, double SL)
  {
   for(int i=PositionsTotal()-1; i>=0; i--)
     {
      if(pos.SelectByIndex(i))
        {
        if(pos.Symbol()==_Symbol)
         {
         if(pos.PositionType()==POSITION_TYPE_BUY)
           {

            if(pos.Profit()>TP || pos.Profit()<SL)
              {
               trade.PositionClose(pos.Ticket());
               Comment("closing Buy Position");
              }
           }
          }
        }
     }

  }
  
  void Close()
  {
   for(int i=PositionsTotal()-1; i>=0; i--)
     {
      if(pos.SelectByIndex(i))
        {
         if(pos.Symbol()==_Symbol)
         {
            printf("Closing Position  with profit"+pos.Profit());
           trade.PositionClose(pos.Ticket());
           
          }

        }
     }

  }



  double RSI(int period){
   double MACDArray[];
   ArraySetAsSeries(MACDArray,true);
 
   int MACDDefinition = iRSI(_Symbol,PERIOD_CURRENT,period, PRICE_CLOSE);
   
   CopyBuffer(MACDDefinition,0,0,3,MACDArray);
   return MACDArray[0];
   
 }
 
double Volume(){
   double MACDArray[];
   ArraySetAsSeries(MACDArray,true);
 
   int MACDDefinition = iVolumes(_Symbol,PERIOD_CURRENT, VOLUME_TICK);
   
   CopyBuffer(MACDDefinition,0,0,3,MACDArray);
   return MACDArray[0];
   
 }

 
 double getProfit()
  {
  double profit=0;
   for(int i=PositionsTotal()-1; i>=0; i--)
     {
      if(pos.SelectByIndex(i))
        {
         if(pos.Symbol()==_Symbol)
         {
          profit= pos.Profit();
         }
         
        }
      }
    return profit;
   }
   
double Momentum(int period){
   double priceArray[];
   int IMOM = iMomentum(_Symbol, PERIOD_M1,period,PRICE_CLOSE);
   CopyBuffer(IMOM,0,0,3,priceArray);
   return priceArray[0];
   
}


double CCI(int period){
   double priceArray[];
   int ICCI = iCCI(_Symbol, PERIOD_M1,period,PRICE_CLOSE);
   CopyBuffer(ICCI,0,0,3,priceArray);
   return priceArray[0];
   
}

double LarryWilliams(int period){
   double priceArray[];
   int IMOM = iWPR(_Symbol, PERIOD_M1,period);
   CopyBuffer(IMOM,0,0,3,priceArray);
   return priceArray[0];
   
}

double SAR(){
   double priceArray[];
   int IMOM = iSAR(_Symbol,PERIOD_CURRENT,0.02,0.2);
   CopyBuffer(IMOM,0,0,3,priceArray);
   return priceArray[0];
   
}
double AO(){
   double priceArray[];
   int IMOM = iAO(_Symbol,PERIOD_CURRENT);
   CopyBuffer(IMOM,0,0,3,priceArray);
   return priceArray[0];
   
}
double BearsPower(int period){
   double priceArray[];
   int IMOM = iBearsPower(_Symbol,PERIOD_CURRENT,period);
   CopyBuffer(IMOM,0,0,3,priceArray);
   return priceArray[0];
   
}
double BullPower(int period){
   double priceArray[];
   int IMOM = iBullsPower(_Symbol,PERIOD_CURRENT,period);
   CopyBuffer(IMOM,0,0,3,priceArray);
   return priceArray[0];
   
}
double RVI(int period){
   double priceArray[];
   int IMOM = iRVI(_Symbol,PERIOD_CURRENT,period);
   CopyBuffer(IMOM,0,0,3,priceArray);
   return priceArray[0];
   
}
void ADX(int timePeriod, double& data[]){
   double MiddelBrandArray[];
   double UpperBandArray[];
   
   
   ArraySetAsSeries(MiddelBrandArray,true);
   ArraySetAsSeries(UpperBandArray,true);
   
   
   int Definition = iADX(_Symbol,PERIOD_CURRENT,timePeriod);
  
   
   CopyBuffer(Definition,0,0,3,MiddelBrandArray);
   CopyBuffer(Definition,1,0,3,UpperBandArray);
   
   
   double MiddelBrandValue=MiddelBrandArray[0];
   double UpperBrandValue = UpperBandArray[0];
   
   
   data[0]=MiddelBrandArray[0];
   data[1]=UpperBandArray[0];
   
   
 }