import random


class TransactionFee(object):
    def __init__(self, amount, fee, is_percentage=False):
        self.amount = amount
        self.fee = fee
        self.is_percentage = is_percentage


class StockSimulationPrice(object):
    def __init__(self, ask_price, bid_price, price):
        self.ask_price = ask_price  # sell
        self.bid_price = bid_price  # buy
        self.price = price


class StockDailySimulationPrices(object):
    def __init__(self, company_id, open_price, close_price, high_price, low_price):
        self.company_id = company_id
        self.open_price = open_price
        self.close_price = close_price
        self.high_price = high_price
        self.low_price = low_price
        self.offset = 0
        self.current_index = 0
        self.simulation_prices = []
        first_price = StockSimulationPrice(open_price, open_price, open_price)
        self.simulation_prices.append(first_price)

    def get_next_prices(self):
        if self.current_index <= self.offset:
            return self.simulation_prices[0]
        count = len(self.simulation_prices)
        index = self.current_index - self.offset
        if index > count - 1:
            index = count - 1
        ret_price = self.simulation_prices[index]
        self.current_index += 1
        return ret_price

    def init_simulation_prices(self, arr):
        for data in arr:
            (normalized_ask_price, normalized_bid_price, normalized_stock_price) = data
            ask_price = round(self.high_price * normalized_ask_price, 3)
            bid_price = round(self.high_price * normalized_bid_price, 3)
            price = round(self.high_price * normalized_stock_price, 3)
            first_price = StockSimulationPrice(ask_price, bid_price, price)
            self.simulation_prices.append(first_price)

        last_price = StockSimulationPrice(self.close_price, self.close_price, self.close_price)
        self.simulation_prices.append(last_price)
        count = len(self.simulation_prices)
        empty_count = max(24 - count - 1, 0)
        if empty_count > 0:
            self.offset = random.randint(1, empty_count)
