//+------------------------------------------------------------------+
//|                                                    dataSaver.mq5 |
//|                        Copyright 2021, MetaQuotes Software Corp. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2021, MetaQuotes Software Corp."
#property link      "https://www.mql5.com"
#property version   "1.00"
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
#include <Indicator.mqh>
double static data[999999][62];
int i =0;
int filecount=0;
//+------------------------------------------------------------------+
//|                                                                  |
//+------------------------------------------------------------------+
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
save();
  }
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
//---
   
   data[i][0]=0;
   data[i][1]=NormalizeDouble(SymbolInfoDouble(_Symbol, SYMBOL_BID),_Digits);
   data[i][2]=NormalizeDouble(SymbolInfoDouble(_Symbol, SYMBOL_ASK),_Digits);
   data[i][3] = iOpen(Symbol(),1,1);
   data[i][4] = iClose(Symbol(),1,1);
   data[i][5] = iHigh(Symbol(),1,1);
   data[i][6] = iLow(Symbol(),1,1);
   data[i][7] = SMA(5,0);
   data[i][8] = SMA(8,0);
   data[i][9] = SMA(13,0);
   data[i][10] = SMA(50,0);
   data[i][11] = SMA(200,0);
   data[i][12] = EMA(8,0);
   data[i][13] = EMA(13,0);
   data[i][14] = EMA(21,0);
   data[i][15] = EMA(50,0);
   data[i][16] = EMA(150,0);
   data[i][17] = EMA5MIN(8,0);
   data[i][18] = EMA5MIN(13,0);
   data[i][19] = EMA5MIN(21,0);
   data[i][20] = STOCHASTIC(5,3,3);
   data[i][21] = RSI(7);
   data[i][22] = RSI(14);
   data[i][23] = RSI(21);
   data[i][24] = Volume();
   data[i][25] = Momentum(7);
   data[i][26] = Momentum(14);
   data[i][27] = Momentum(21);
   double BB[3];
   BollingerBrand(21,0,2, BB);
   data[i][28] = BB[0];
   data[i][29] = BB[1];
   data[i][30] = BB[2];
   BollingerBrand(14,0,2, BB);
   data[i][31] = BB[0];
   data[i][32] = BB[1];
   data[i][33] = BB[2];
   data[i][34] = CCI(7);
   data[i][35] = CCI(14);
   data[i][36] = CCI(21);
   data[i][37] = 0;
   data[i][38] = 0;
   data[i][39] = 0;
   data[i][40] = 0;
   data[i][41] = LarryWilliams(21);
   data[i][42] = LarryWilliams(14);
   data[i][43] = LarryWilliams(7);
   data[i][44] = SAR();
   data[i][45] = AO();
   data[i][46] = BearsPower(21);
   data[i][47] = BearsPower(14);
   data[i][48] = BearsPower(7);
   data[i][49] = BearsPower(21);
   data[i][50] = BearsPower(14);
   data[i][51] = BearsPower(7);
   data[i][52] = RVI(21);
   data[i][53] = RVI(14);
   data[i][54] = RVI(7);
   double Adx[2];
   ADX(21,Adx);
   data[i][55] = Adx[0];
   data[i][56] = Adx[1];
   ADX(14,Adx);
   data[i][57] = Adx[0];
   data[i][58] = Adx[1];
   ADX(7,Adx);
   data[i][59] = Adx[0];
   data[i][60] = Adx[1];
   data[i][61]=0;
   i=i+1;
   
   if(i==999999){
   save();
   i=0;
   }
   
   
   
  }
//+------------------------------------------------------------------+
void save(){
string filename = _Symbol+"data"+filecount+".csv";
   int fileHandler = FileOpen(filename, FILE_READ|FILE_WRITE|FILE_CSV,",");
     {
     printf(TerminalInfoString(TERMINAL_COMMONDATA_PATH));
     for(int j=0;j<i;j++){
     FileSeek(fileHandler,0,SEEK_END);
     
      FileWrite(fileHandler, data[j][0], data[j][1], data[j][2], data[j][3], data[j][4], data[j][5], data[j][6], data[j][7], data[j][8], data[j][9], data[j][10], data[j][11], data[j][12], data[j][13], data[j][14], data[j][15], data[j][16],
       data[j][17], data[j][18], data[j][19], data[j][20], data[j][21], data[j][22], data[j][23], data[j][24], data[j][25], data[j][26], data[j][27], data[j][28], data[j][29], data[j][30], data[j][31], data[j][32], data[j][33], data[j][34], data[j][35],
        data[j][36], data[j][37], data[j][38], data[j][39], data[j][40], data[j][41], data[j][42], data[j][43], data[j][44], data[j][45], data[j][46], data[j][47], data[j][48], data[j][49], data[j][50], data[j][51], data[j][52], data[j][53], data[j][54],
         data[j][55], data[j][56], data[j][57], data[j][58], data[j][59], data[j][60], data[j][61]);
     
     }
      

      FileClose(fileHandler);
      printf("SAVE FILE :"+filecount);
      filecount=filecount+1;
     }




}