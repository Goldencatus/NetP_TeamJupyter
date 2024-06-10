from datetime import datetime,timedelta
import pandas as pd
import requests
from typing import *
import time
import numpy as np

class BinanceClient:
    def __init__(self, futures=False):
        self.exchange = "BINANCE"
        self.futures = futures

        if self.futures:
            self._base_url = "https://fapi.binance.com"
        else:
            self._base_url = "https://api.binance.com"

        self.symbols = self._get_symbols()

    def _make_request(self, endpoint: str, query_parameters: Dict):
        try:
            response = requests.get(self._base_url + endpoint, params=query_parameters)
        except Exception as e:
            print("Connection error while making request to %s: %s", endpoint, e)
            return None

        if response.status_code == 200:
            return response.json()
        else:
            print("Error while making request to %s: %s (status code = %s)",
                         endpoint, response.json(), response.status_code)
            return None

    def _get_symbols(self) -> List[str]:

        params = dict()

        endpoint = "/fapi/v1/exchangeInfo" if self.futures else "/api/v3/exchangeInfo"
        data = self._make_request(endpoint, params)

        symbols = [x["symbol"] for x in data["symbols"]]

        return symbols

    def get_historical_data(self, symbol: str, interval: Optional[str] = "1h", start_time: Optional[int] = None, end_time: Optional[int] = None, limit: Optional[int] = 1500):

        params = dict()

        params["symbol"] = symbol
        params["interval"] = interval
        params["limit"] = limit

        if start_time is not None:
            params["startTime"] = start_time
        if end_time is not None:
            params["endTime"] = end_time

        endpoint = "/fapi/v1/klines" if self.futures else "/api/v3/klines"
        raw_candles = self._make_request(endpoint, params)

        candles = []

        if raw_candles is not None:
            for c in raw_candles:
                candles.append((float(c[0]), float(c[1]), float(c[2]), float(c[3]), float(c[4]), float(c[5]),))
            return candles
        else:
            return None


class DataSource:
    def __init__(self,trend_length,cnn_length,lstm_length,symbol,start_time,end_time,interval,batch_size):
        self.client = BinanceClient(futures=False)
        self.trend_length = trend_length
        self.cnn_length = cnn_length
        self.lstm_length = lstm_length
        self.symbol = symbol
        self.batch_size= batch_size
        self.start_time = start_time
        self.end_time = end_time
        self.interval = interval
        self.fromDate = int((datetime.strptime(start_time, '%Y-%m-%d')-timedelta(hours=cnn_length+lstm_length-2)).timestamp() * 1000)
        self.toDate = int((datetime.strptime(end_time, '%Y-%m-%d')-timedelta(hours=1)+timedelta(hours=trend_length-1)).timestamp() * 1000)
        self.stock_data = self.GetDataFrame(self.GetHistoricalData())
        self.stock_data["Label"] = self.stock_data["Close"].shift(-self.trend_length)
        self.stock_data=self.stock_data.dropna(axis = 0)
        '''
        self.stock_data["Label"] = self.stock_data["Label"] - self.stock_data["Close"]
        self.stock_data["Label_neg"]  = self.stock_data["Label"] <= 0
        self.stock_data["Label_pos"]  = self.stock_data["Label"] > 0
        self.label = self.stock_data[["Label_neg","Label_pos"]].to_numpy()
        '''
        self.label = self.stock_data["Label"]
        self.label = self.label[cnn_length+lstm_length-2:]
        self.np_cnn_data = self.stock_data[["Open", "High", "Low", "Close","Volume"]].to_numpy()
        #Batch normalization
        #self.np_cnn_data = (self.np_cnn_data - self.np_cnn_data.mean(axis=0)) / self.np_cnn_data.std(axis=0)
        
        self.np_lstm_data = self.np_cnn_data[self.cnn_length-1:].copy()
        self.cnn_data_temp = []
        self.cnn_data = []
        self.lstm_data = []
        self.label_data = []
        label_index = 0
        while(label_index + self.batch_size <= len(self.label)):
            self.label_data.append(self.label[label_index:label_index+self.batch_size].copy())
            label_index += self.batch_size
        self.label_data.append(self.label[label_index:].copy())


        for i in range(len(self.np_cnn_data)-self.cnn_length+1):
            self.cnn_data_temp.append(self.np_cnn_data[i:i+self.cnn_length].copy())
        for i in range(len(self.cnn_data_temp)-self.lstm_length+1):
            if(i%self.batch_size == 0):
                self.cnn_data.append([])
            self.cnn_data[-1].append(self.cnn_data_temp[i:i+self.lstm_length].copy())
        for i in range(len(self.cnn_data)):
            self.cnn_data[i] = np.array(self.cnn_data[i])

        for i in range(len(self.np_lstm_data)-self.lstm_length+1):
            if(i%self.batch_size == 0):
                self.lstm_data.append([])
            self.lstm_data[-1].append(self.np_lstm_data[i:i+self.lstm_length].copy())
        for i in range(len(self.lstm_data)):
            self.lstm_data[i] = np.array(self.lstm_data[i])
        
    
    
        
    def ms_to_dt_utc(self,ms: int) -> datetime:
        return datetime.utcfromtimestamp(ms / 1000)

    def ms_to_dt_local(self,ms: int) -> datetime:
        return datetime.fromtimestamp(ms / 1000)

    def GetDataFrame(self,data):
        df = pd.DataFrame(data, columns=['Timestamp', "Open", "High", "Low", "Close", "Volume"])
        df["Timestamp"] = df["Timestamp"].apply(lambda x: self.ms_to_dt_local(x))
        df['Date'] = df["Timestamp"].dt.strftime("%d/%m/%Y")
        df['Time'] = df["Timestamp"].dt.strftime("%H:%M:%S")
        column_names = ["Date", "Time","Open", "High", "Low", "Close", "Volume"]
        df = df.set_index('Timestamp')
        df = df.reindex(columns=column_names)

        return df

    def GetHistoricalData(self):
        collection = []
        start_time = self.fromDate
        end_time = self.toDate
        while start_time < end_time:
            data = self.client.get_historical_data(self.symbol, start_time=start_time, end_time=end_time,interval= self.interval)
            start_time = int(data[-1][0] + 1000)
            collection +=data
            time.sleep(1.1)

        return collection
