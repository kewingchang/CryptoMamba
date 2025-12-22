

def calculate_trade_setup(entry_price, signal_direction, atr, mode='aggressive', ema7=None, ema14=None):
    """
    根据 ATR、EMA 和 信号强弱模式 计算挂单策略 (Fusion Strategy)
    
    Args:
        entry_price: 当前开盘价
        signal_direction: 'LONG' or 'SHORT'
        atr: ATR_14 值
        mode: 'aggressive' (强信号) 或 'conservative' (弱信号)
        ema7: EMA_7 价格
        ema14: EMA_14 价格
    """
    orders = []
    stop_loss = 0
    
    # 辅助函数：计算做多时的挂单价 (支撑位)
    # 在 ATR 计算出的支撑位和 EMA 均线之间，选那个更高的价格挂单（防止踏空均线）
    # 前提是 EMA 必须在现价下方（有效支撑），如果在现价上方则是阻力，忽略之
    def get_long_limit_price(atr_target, ema_ref):
        if ema_ref is not None and ema_ref < entry_price:
            return max(atr_target, ema_ref)
        return atr_target

    # 辅助函数：计算做空时的挂单价 (阻力位)
    # 在 ATR 计算出的阻力位和 EMA 均线之间，选那个更低的价格挂单
    def get_short_limit_price(atr_target, ema_ref):
        if ema_ref is not None and ema_ref > entry_price:
            return min(atr_target, ema_ref)
        return atr_target

    # === 策略参数配置 ===
    if mode == 'aggressive':
        # 强信号：30% 市价 + 30% 接EMA7/0.3ATR + 40% 接EMA14/0.6ATR
        atr_stop_mult = 1.0
        
        # 订单结构: type, size, atr_offset, ema_reference
        setup = [
            {'type': 'market', 'size': 0.3, 'offset': 0.0, 'ema': None},
            {'type': 'limit',  'size': 0.3, 'offset': 0.3, 'ema': ema7},  # 第一补仓参考 EMA7
            {'type': 'limit',  'size': 0.4, 'offset': 0.6, 'ema': ema14}  # 第二补仓参考 EMA14
        ]
        
    elif mode == 'conservative':
        # 弱信号：无市价 + 50% 接EMA14/0.5ATR + 50% 深针0.8ATR
        atr_stop_mult = 1.0
        
        setup = [
            {'type': 'limit', 'size': 0.3, 'offset': 0.3, 'ema': ema7},
            {'type': 'limit', 'size': 0.3, 'offset': 0.6, 'ema': ema14},
            {'type': 'limit', 'size': 0.4, 'offset': 0.8, 'ema': None}
        ]
    else:
        return [], 0

    # === 生成订单 ===
    if signal_direction == "LONG":
        stop_loss = entry_price - (atr * atr_stop_mult)
        
        for item in setup:
            # 计算纯 ATR 目标价
            raw_atr_price = entry_price - (atr * item['offset'])
            
            if item['type'] == 'market':
                final_price = entry_price
            else:
                # 融合 EMA 逻辑
                final_price = get_long_limit_price(raw_atr_price, item['ema'])
            
            orders.append({'type': item['type'], 'size': item['size'], 'price': final_price})
            
    elif signal_direction == "SHORT":
        stop_loss = entry_price + (atr * atr_stop_mult)
        
        for item in setup:
            raw_atr_price = entry_price + (atr * item['offset'])
            
            if item['type'] == 'market':
                final_price = entry_price
            else:
                # 融合 EMA 逻辑
                final_price = get_short_limit_price(raw_atr_price, item['ema'])
                
            orders.append({'type': item['type'], 'size': item['size'], 'price': final_price})
            
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