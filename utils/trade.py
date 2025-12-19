# 示例调用
# orders, sl = calculate_trade_setup(2826, "SHORT", 185)
# print(f"Stop Loss: {sl}") 
# Result: Stop Loss 3011. Orders at 2826, 2881.5, 2937
def calculate_trade_setup(entry_price, signal_direction, atr, leverage=2):
    """
    根据 ATR 计算 DCA 挂单点和止损点
    Strategy: 30% Market, 30% @0.3ATR, 40% @0.6ATR. Stop @1.0ATR
    """
    atr_Stop_Multiplier = 1.0
    dca_1_Multiplier = 0.3
    dca_2_Multiplier = 0.6
    
    orders = []
    
    if signal_direction == "LONG":
        stop_loss = entry_price - (atr * atr_Stop_Multiplier)
        
        # Order 1: Market
        orders.append({'type': 'market', 'size': 0.3, 'price': entry_price})
        # Order 2: Limit Buy (DCA 1)
        orders.append({'type': 'limit', 'size': 0.3, 'price': entry_price - (atr * dca_1_Multiplier)})
        # Order 3: Limit Buy (DCA 2)
        orders.append({'type': 'limit', 'size': 0.4, 'price': entry_price - (atr * dca_2_Multiplier)})
        
    elif signal_direction == "SHORT":
        stop_loss = entry_price + (atr * atr_Stop_Multiplier)
        
        # Order 1: Market
        orders.append({'type': 'market', 'size': 0.3, 'price': entry_price})
        # Order 2: Limit Sell (DCA 1)
        orders.append({'type': 'limit', 'size': 0.3, 'price': entry_price + (atr * dca_1_Multiplier)})
        # Order 3: Limit Sell (DCA 2)
        orders.append({'type': 'limit', 'size': 0.4, 'price': entry_price + (atr * dca_2_Multiplier)})   
    return orders, stop_loss

def buy_sell_smart(today, pred, balance, shares, risk=5):
    diff = pred * risk / 100
    if today > pred + diff:
        balance += shares * today
        shares = 0
    elif today > pred:
        factor = (today - pred) / diff
        balance += shares * factor * today
        shares *= (1 - factor)
    elif today > pred - diff:
        factor = (pred - today) / diff
        shares += balance * factor / today
        balance *= (1 - factor)
    else:
        shares += balance / today
        balance = 0
    return balance, shares

def buy_sell_smart_w_short(today, pred, balance, shares, risk=5, max_n_btc=0.002):
    diff = pred * risk / 100
    if today < pred - diff:
        shares += balance / today
        balance = 0
    elif today < pred:
        factor = (pred - today) / diff
        shares += balance * factor / today
        balance *= (1 - factor)
    elif today < pred + diff:
        if shares > 0:
            factor = (today - pred) / diff
            balance += shares * factor * today
            shares *= (1 - factor)
    else:
        balance += (shares + max_n_btc) * today
        shares = -max_n_btc
    return balance, shares

def buy_sell_vanilla(today, pred, balance, shares, tr=0.01):
    tmp = abs((pred - today) / today)
    if tmp < tr:
        return balance, shares
    if pred > today:
        shares += balance / today
        balance = 0
    else:
        balance += shares * today
        shares = 0
    return balance, shares


def trade(data, time_key, timstamps, targets, preds, balance=100, mode='smart_v2', risk=5, y_key='Close'):
    balance_in_time = [balance]
    shares = 0

    for ts, target, pred in zip(timstamps, targets, preds):
        today = data[data[time_key] == int(ts - 24 * 60 * 60)].iloc[0][y_key]
        assert round(target, 2) == round(data[data[time_key] == int(ts)].iloc[0][y_key], 2)
        if mode == 'smart':
            balance, shares = buy_sell_smart(today, pred, balance, shares, risk=risk)
        if mode == 'smart_w_short':
            balance, shares = buy_sell_smart_w_short(today, pred, balance, shares, risk=risk, max_n_btc=0.002)
        elif mode == 'vanilla':
            balance, shares = buy_sell_vanilla(today, pred, balance, shares)
        elif mode == 'no_strategy':
            shares += balance / today
            balance = 0
        balance_in_time.append(shares * today + balance)

    balance += shares * targets[-1]
    return balance, balance_in_time