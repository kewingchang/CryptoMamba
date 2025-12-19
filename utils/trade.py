
def calculate_trade_setup(entry_price, signal_direction, atr, mode='aggressive'):
    """
    根据 ATR 和 信号强弱模式 计算挂单策略
    
    Args:
        mode (str): 'aggressive' (强信号) 或 'conservative' (弱信号)
    """
    orders = []
    stop_loss = 0
    
    # === 策略参数配置 ===
    if mode == 'aggressive':
        # 强信号：怕踏空，先买一部分，防画门
        # 仓位: 30% 市价, 30% @-0.3ATR, 40% @-0.6ATR
        # 止损: 1.0 ATR
        atr_stop_mult = 1.0
        levels = [
            {'type': 'market', 'size': 0.3, 'offset': 0.0},
            {'type': 'limit',  'size': 0.3, 'offset': 0.3},
            {'type': 'limit',  'size': 0.4, 'offset': 0.6}
        ]
        
    elif mode == 'conservative':
        # 弱信号：预测涨幅小，只接针，不追高
        # 仓位: 0% 市价, 50% @-0.5ATR, 50% @-0.8ATR (更深的位置)
        # 止损: 1.2 ATR (给更多空间，因为入场点已经很低了)
        atr_stop_mult = 1.0
        levels = [
            # 注意：没有 market 单
            {'type': 'limit', 'size': 0.5, 'offset': 0.5},
            {'type': 'limit', 'size': 0.5, 'offset': 0.8}
        ]
    else:
        return [], 0

    # === 生成订单 ===
    if signal_direction == "LONG":
        stop_loss = entry_price - (atr * atr_stop_mult)
        for lvl in levels:
            price = entry_price if lvl['type'] == 'market' else entry_price - (atr * lvl['offset'])
            orders.append({'type': lvl['type'], 'size': lvl['size'], 'price': price})
            
    elif signal_direction == "SHORT":
        stop_loss = entry_price + (atr * atr_stop_mult)
        for lvl in levels:
            price = entry_price if lvl['type'] == 'market' else entry_price + (atr * lvl['offset'])
            orders.append({'type': lvl['type'], 'size': lvl['size'], 'price': price})
            
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