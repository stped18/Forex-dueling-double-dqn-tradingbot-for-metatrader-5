

import MetaTrader5 as mt5


def Buy_Action(mt, symbol):
    print("Buy")
    symbol_info = mt.symbol_info(symbol)
    if symbol_info is None:
        print(symbol, "not found, can not call order_check()")
        return

    if not symbol_info.visible:
        print(symbol, "is not visible, trying to switch on")
        if not mt.symbol_select(symbol, True):
            print("symbol_select({}}) failed, exit", symbol)
            return
    point = mt.symbol_info(symbol).point
    price = mt.symbol_info_tick(symbol).ask
    deviation = 50
    lot=0.01
    request = {
        "action": mt.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot,
        "type": mt.ORDER_TYPE_BUY,
        "price": price,
        #"sl": 0,
        #"tp": 0,
        "deviation": deviation,
        "magic": 234000,
        "comment": "python script open",
        "type_time": mt.ORDER_TIME_GTC,
        "type_filling": mt.ORDER_FILLING_FOK,
    }

    result = mt.order_send(request)
    print("1. order_send(): by {} {} lots at {} with deviation={} points".format(symbol, lot, price, deviation))
    if result.retcode != mt.TRADE_RETCODE_DONE:
        print("2. order_send failed, retcode={}".format(result.retcode))
        # request the result as a dictionary and display it element by element
        result_dict = result._asdict()
        for field in result_dict.keys():
            print("   {}={}".format(field, result_dict[field]))
            # if this is a trading request structure, display it element by element as well
            if field == "request":
                traderequest_dict = result_dict[field]._asdict()
                for tradereq_filed in traderequest_dict:
                    print("       traderequest: {}={}".format(tradereq_filed, traderequest_dict[tradereq_filed]))
    print("2. order_send done, ", result)
    print("   opened position with POSITION_TICKET={}".format(result.order))
    return {"request": request,
            "result": result}

def Sell_Avtion(mt, symbol):
    print("Sell")
    symbol_info = mt.symbol_info(symbol)
    if symbol_info is None:
        print(symbol, "not found, can not call order_check()")
        return
    if not symbol_info.visible:
        print(symbol, "is not visible, trying to switch on")
        if not mt.symbol_select(symbol, True):
            print("symbol_select({}}) failed, exit", symbol)
            return
    point = mt.symbol_info(symbol).point
    price = mt.symbol_info_tick(symbol).bid
    deviation = 50
    lot=0.01
    request = {
        "action": mt.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot,
        "type": mt.ORDER_TYPE_SELL,
        "price": price,
        #"sl": 0,
        #"tp": 0,
        "deviation": deviation,
        "magic": 234000,
        "comment": "python script open",
        "type_time": mt.ORDER_TIME_GTC,
        "type_filling": mt.ORDER_FILLING_FOK,
    }
    result = mt.order_send(request)
    print("1. order_send(): by {} {} lots at {} with deviation={} points".format(symbol, lot, price, deviation))
    if result.retcode != mt.TRADE_RETCODE_DONE:
        print("2. order_send failed, retcode={}".format(result.retcode))
        # request the result as a dictionary and display it element by element
        result_dict = result._asdict()
        for field in result_dict.keys():
            print("   {}={}".format(field, result_dict[field]))
            # if this is a trading request structure, display it element by element as well
            if field == "request":
                traderequest_dict = result_dict[field]._asdict()
                for tradereq_filed in traderequest_dict:
                    print("       traderequest: {}={}".format(tradereq_filed, traderequest_dict[tradereq_filed]))
    print("2. order_send done, ", result)
    print("   opened position with POSITION_TICKET={}".format(result.order))
    return {"request": request,
            "result": result}
def Close_Position(mt, symbol, result , action):
    if action == 'short':
        trade_type = mt.ORDER_TYPE_BUY
        price = mt.symbol_info_tick(symbol).ask
    else:
        trade_type = mt.ORDER_TYPE_SELL
        price = mt.symbol_info_tick(symbol).bid
    position_id = result.order
    deviation = 50
    lot=0.01
    request = {
        "action": mt.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot,
        "type": trade_type,
        "position": position_id,
        "price": price,
        "deviation": deviation,
        "magic": 234000,
        "comment": "python script close",
        "type_time": mt.ORDER_TIME_GTC,
        "type_filling": mt.ORDER_FILLING_RETURN,
    }
    # send a trading request
    result = mt.order_send(request)
    # check the execution result
    print(
        "3. close position #{}: sell {} {} lots at {} with deviation={} points".format(position_id, symbol, lot, price,
                                                                                       deviation));
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print("4. order_send failed, retcode={}".format(result.retcode))
        print("   result", result)
    else:
        print("4. position #{} closed, {}".format(position_id, result))
        # request the result as a dictionary and display it element by element
        result_dict = result._asdict()
        for field in result_dict.keys():
            print("   {}={}".format(field, result_dict[field]))
            # if this is a trading request structure, display it element by element as well
            if field == "request":
                traderequest_dict = result_dict[field]._asdict()
                for tradereq_filed in traderequest_dict:
                    print("       traderequest: {}={}".format(tradereq_filed, traderequest_dict[tradereq_filed]))