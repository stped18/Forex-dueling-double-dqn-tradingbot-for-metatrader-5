

class Order(object):
    def __init__(self, ask, bid, position):
        self.ask=ask
        self.bid=bid
        self.position = position
        self.profit=0
        self.close_position=False
        self.close_Price=0

    def calculate_prifit(self, new_ask, new_bid):
        if self.position=="long":
            self.profit= new_bid-self.ask
        if self.position=="short":
            self.profit=new_ask-self.bid
        return self.profit

    def close_Position(self, new_ask, new_bid ):
        if self.position=="long":
            self.close_Price=new_bid
            self.profit = new_bid - self.ask
        if self.position=="short":
            self.close_Price=new_ask
            self.profit = new_ask - self.bid
        self.close_position = True

