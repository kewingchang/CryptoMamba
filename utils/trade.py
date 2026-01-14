
def get_trade_decision(symbol, x_factor, pred_direction):
    """
    根据品种和信号强度，决定怎么做
    """
    decision = 'WAIT'
    mode = 'conservative'
    inverse = False  # 是否反向交易
    
    if symbol == 'BTC':
        if 0.0 <= x_factor < 0.3:
            decision = 'OPEN'
            mode = 'conservative' # 震荡区，挂单接
        elif 0.9 <= x_factor < 1.0:
            decision = 'OPEN'
            mode = 'aggressive'   # 黄金区，追单
        elif x_factor >= 1.0:
            decision = 'OPEN'
            mode = 'aggressive'
            inverse = True        # 反向开单！
            
    elif symbol == 'ETH':
        if 0.0 <= x_factor < 0.4:
            decision = 'OPEN'
            mode = 'conservative'
        elif x_factor >= 0.8:     # 包括了 0.8-0.9, 0.9-1.0 和 >1.0
            decision = 'OPEN'
            mode = 'aggressive'   # 趋势区，追单
            
    # 执行反向逻辑
    final_direction = pred_direction
    if inverse:
        final_direction = 'SHORT' if pred_direction == 'LONG' else 'LONG'
        print(f"⚠️ 触发反向策略！模型预测 {pred_direction}, 实际执行 {final_direction}")
        print("⚠️ 只下单DCA2, DCA3")
        
    return decision, mode, final_direction


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