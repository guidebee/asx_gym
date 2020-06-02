import random

HOLD_STOCK = 0
BUY_STOCK = 1
SELL_STOCK = 2
TOP_UP_FUND = 3
WITHDRAW_FUND = 4


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

    def to_json_obj(self):
        if self.stock_operation == BUY_STOCK:
            stock_operation = 'buy'
        elif self.stock_operation == SELL_STOCK:
            stock_operation = 'sell'
        elif self.stock_operation == HOLD_STOCK:
            stock_operation = 'hold'
        elif self.stock_operation == TOP_UP_FUND:
            stock_operation = 'top_up'
        elif self.stock_operation == WITHDRAW_FUND:
            stock_operation = 'withdraw'
        json_obj = {
            'company_id': int(self.company_id),
            'stock_operation': stock_operation,
            'volume': round(float(self.volume), 2),
            'price': round(float(self.price), 2)
        }
        return json_obj


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

    def to_json_obj(self):
        json_obj = {
            'end_batch': int(self.end_batch),
            'transactions': []
        }
        for transaction in self.transactions:
            json_obj['transactions'].append(transaction.to_json_obj())
        return json_obj

    @staticmethod
    def from_env_action(action):
        company_count = action['company_count']
        end_batch = action['end_batch']
        asx_action = AsxAction(end_batch)
        for c in range(company_count):
            company_id = action['company_id'][c]
            volume = action['volume'][c]
            price = action['price'][c]
            stock_operation = action['stock_operation'][c]
            asx_transaction = AsxTransaction(company_id, stock_operation, volume, price)
            asx_action.add_transaction(asx_transaction)
        return asx_action


class StockIndex:
    def __init__(self, index_date, open_index, close_index, high_index, low_index):
        self.index_date = index_date
        self.open_index = open_index
        self.close_index = close_index
        self.high_index = high_index
        self.low_index = low_index

    def to_json_obj(self):
        json_obj = {
            'index_date': self.index_date,
            'open_index': round(self.open_index, 2),
            'close_index': round(self.close_index, 2),
            'high_index': round(self.high_index, 2),
            'low_index': round(self.low_index, 2)

        }
        return json_obj


class StockPrice:
    def __init__(self, price_date, company_id, open_price,
                 close_price, high_price, low_price):
        self.price_date = price_date
        self.company_id = company_id
        self.open_price = open_price
        self.close_price = close_price
        self.high_price = high_price
        self.low_price = low_price

    def to_json_obj(self):
        json_obj = {
            'price_date': self.price_date,
            'company_id': int(self.company_id),
            'open_price': round(self.open_price, 2),
            'close_price': round(self.close_price, 2),
            'high_price': round(self.high_price, 2),
            'low_price': round(self.low_price, 2)
        }
        return json_obj


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
        json_obj = {"day": int(self.day),
                    "seconds": int(self.seconds),
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
