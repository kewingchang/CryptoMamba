
def get_trade_decision(symbol, x_factor, pred_direction):
    """
    基于回测数据的精细化决策逻辑
    
    Args:
        symbol: 'BTC-USD' or 'ETH-USD'
        x_factor: 信号强度
        pred_direction: 'LONG' or 'SHORT'
        
    Returns:
        decision: 'OPEN' or 'WAIT'
        mode: 'MARKET' or 'LIMIT'
        final_direction: 实际执行方向
    """
    decision = 'WAIT'
    mode = 'LIMIT'
    inverse = False
    
    # 提取基础币种名称 (兼容 BTC-USD, BTCUSDT 等)
    base_symbol = symbol
    
    if base_symbol == 'BTC':
        # BTC 策略: 避坑 + 反向
        if 0.0 <= x_factor < 0.1:
            # 胜率 60%，收益 +14.8%
            decision = 'OPEN'
            mode = 'LIMIT' # 信号太弱，不追市价
            
        elif 0.1 <= x_factor < 0.2:
            # 胜率 33%，收益 -15.4% -> 坚决反向！
            decision = 'OPEN'
            mode = 'LIMIT' # 反向后变为顺势，但因波动率低，仍保守
            inverse = True
            
        elif 0.2 <= x_factor < 0.3:
            # 胜率 69%，收益 +7.1% -> 黄金区间
            decision = 'OPEN'
            mode = 'MARKET'   # 胜率极高，值得追市价
            
        else:
            # > 0.3 整体表现不佳，放弃
            decision = 'WAIT'
            
    elif base_symbol == 'ETH':
        # ETH 策略: 顺势而为
        if 0.0 <= x_factor < 0.1:
            # 0.0-0.1 还可以 -> 整体保守做
            decision = 'OPEN'
            mode = 'LIMIT'

        elif 0.1 <= x_factor < 0.2:
            # 0.1-0.2 微亏 -> 整体保守做
            decision = 'OPEN'
            mode = 'LIMIT - DCA2, DCA3 ONLY'

        elif 0.2 <= x_factor < 0.5:
            # 黄金爆发区！0.4-0.5 收益高达 20%
            decision = 'OPEN'
            mode = 'LIMIT'
            
        else:
            # > 0.5 开始亏损，放弃
            decision = 'WAIT'
    
    # 执行反向逻辑
    final_direction = pred_direction
    if inverse and decision == 'OPEN':
        final_direction = 'SHORT' if pred_direction == 'LONG' else 'LONG'
        print(f"⚠️ [Strategy] Trigger INVERSE trade for {symbol} (x={round(x_factor,3)})")
        
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