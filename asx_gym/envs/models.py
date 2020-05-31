import random


class TransactionFee:
    def __init__(self, amount, fee, is_percentage=False):
        self.amount = amount
        self.fee = fee
        self.is_percentage = is_percentage


class StockSimulationPrice:
    def __init__(self, ask_price, bid_price, price):
        self.ask_price = ask_price  # sell
        self.bid_price = bid_price  # buy
        self.price = price


class StockDailySimulationPrices:
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
            ret_price = self.simulation_prices[0]
        else:
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


class AsxTransaction:
    def __init__(self, company_id, stock_operation, volume, price):
        self.company_id = company_id
        self.stock_operation = stock_operation
        self.volume = volume
        self.price = price


class AsxAction:
    def __init__(self, end_batch):
        self.end_batch = end_batch
        self.transactions = []

    def add_transaction(self, transaction: AsxTransaction):
        self.transactions.append(transaction)

    def copy_to_env_action(self, action):
        company_count = len(self.transactions)
        action['company_count'] = company_count
        action['end_batch'] = self.end_batch
        for c in range(company_count):
            asx_transaction: AsxTransaction = self.transactions[c]
            action['company_id'][c] = asx_transaction.company_id
            action['volume'][c] = asx_transaction.volume
            action['price'][c] = asx_transaction.price
            action['stock_operation'][c] = asx_transaction.stock_operation
        return action

    @staticmethod
    def from_env_action(action):
        pass


class StockIndex:
    def __init__(self, index_date, open_index, close_index, high_index, low_index):
        self.index_date = index_date
        self.open_index = open_index
        self.close_index = close_index
        self.high_index = high_index
        self.low_index = low_index


class StockPrice:
    def __init__(self, price_date, company_id, open_price,
                 close_price, high_price, low_price):
        self.price_date = price_date
        self.company_id = company_id
        self.open_price = open_price
        self.close_price = close_price
        self.high_price = high_price
        self.low_price = low_price


class StockRecord:
    def __init__(self, company_id, volume, buy_price, sell_price, price):
        self.company_id = company_id
        self.volume = volume
        self.buy_price = buy_price
        self.sell_price = sell_price
        self.price = price


class AsxObservation:
    def __init__(self, observation):
        self.day = observation['day']
        self.seconds = observation['second']
        self.total_value = float(observation['total_value'].item())
        self.available_fund = float(observation['available_fund'].item())
        self.bank_balance = float(observation['bank_balance'].item())
        open_index = float(observation['indexes']['open'].item())
        close_index = float(observation['indexes']['close'].item())
        high_index = float(observation['indexes']['high'].item())
        low_index = float(observation['indexes']['low'].item())
        self.stock_index = StockIndex('', open_index,
                                      close_index, high_index, low_index)

        self.portfolios = []
        self.prices = {}
        company_count = observation['company_count']
        for c in range(company_count):
            company_id = observation['prices']['company_id'][c].item()
            ask_price = observation['prices']['ask_price'][c].item()
            bid_price = observation['prices']['bid_price'][c].item()
            price = observation['prices']['price'][c].item()
            self.prices[company_id] = StockSimulationPrice(ask_price, bid_price, price)

        portfolio_company_count = observation['portfolio_company_count']
        for c in range(portfolio_company_count):
            company_id = observation['portfolios']['company_id'][c].item()
            volume = observation['portfolios']['volume'][c].item()
            buy_price = observation['portfolios']['buy_price'][c].item()
            sell_price = observation['portfolios']['sell_price'][c].item()
            price = observation['portfolios']['price'][c].item()
            stock_record = StockRecord(company_id, volume, buy_price, sell_price, price)
            self.portfolios.append(stock_record)

    def to_json_obj(self):
        json_obj = {"day": self.day,
                    "seconds": self.seconds,
                    "total_value": round(self.total_value, 2),
                    "available_fund": round(self.available_fund, 2),
                    "bank_balance": round(self.bank_balance, 2),
                    "index": {
                        "open": round(self.stock_index.open_index, 2),
                        "close": round(self.stock_index.close_index, 2),
                        "high": round(self.stock_index.high_index, 2),
                        "low": round(self.stock_index.low_index, 2)
                    },
                    "prices": {},
                    "portfolios": {}}
        for company_id, prices in self.prices.items():
            json_obj["prices"][company_id] = {
                "ask_price": round(prices.ask_price, 2),
                "bid_price": round(prices.bid_price, 2),
                "price": round(prices.price, 2)

            }
        for stock_record in self.portfolios:
            json_obj["portfolios"][stock_record.company_id] = {
                "volume": round(stock_record.volume, 2),
                "buy_price": round(stock_record.buy_price, 2),
                "sell_price": round(stock_record.sell_price, 2),
                "price": round(stock_record.price, 2)
            }

        return json_obj
