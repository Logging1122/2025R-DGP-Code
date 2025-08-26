
import numpy as np


class KRR:
    def __init__(self, domain, epsilon):
        # item所在的空间，比如 ['a','b','c']
        self.domain = domain
        self.k = len(domain)
        self.epsilon = epsilon
        # 高概率ph，低概率pl
        self.ph = np.e**epsilon / (np.e**epsilon + self.k - 1)
        self.pl = 1 / (np.e**epsilon + self.k - 1)

        # 用于快速建立索引，比如{'a':0, 'b':1, 'c':2}，以下两种写法相同
        # self.item_index = {item: index for index, item in enumerate(domain)}
        self.item_index = {tuple(item): index for index, item in enumerate(domain)}
        # self.item_index = dict(zip(domain, range(len(domain))))


    def encode(self, item):
        if list(item) not in self.domain:
            raise Exception("ERR: the input is error, item = ", item)
        probability_arr = np.full(shape=self.k, fill_value=self.pl)
        probability_arr[self.item_index[item]] = self.ph
        index=np.random.choice(a=len(self.domain), p=probability_arr)
        return self.domain[index]

        # return np.random.choice(a=self.domain, p=probability_arr)


    def decode_by_item(self, item_list):
        cnt_list = np.zeros(shape=self.k)
        for item in item_list:
            cnt_list[self.item_index[item]] += 1
        esti_cnt = (cnt_list - len(item_list)*self.pl) / (self.ph - self.pl)
        return esti_cnt



# if __name__ == '__main__':
#     # domain=[[1,2],[2,3],[3,4]]
#     domain = [[41.8654847, -87.61699675], [41.9205921899, -87.6374657275], [41.8787000494, -87.6396560669]]
#     epsilon = 1
#     krr = KRR(domain, epsilon)
#     for item in domain:
#         result=krr.encode(tuple(item))
#         print(result)
