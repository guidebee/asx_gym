class TransactionFee(object):
    def __init__(self, amount, fee, is_percentage=False):
        self.amount = amount
        self.fee = fee
        self.is_percentage = is_percentage
