import config
import csv
from binance.client import Client

client = Client(config.API_KEY, config.API_SECRET)

# prices = client.get_all_tickers()

# for price in prices:
#     print(price)


# candles = client.get_klines(symbol="BTCUSDT", interval=Client.KLINE_INTERVAL_1MINUTE)

# csvfile = open('1minutes.csv', 'w', newline='')
# candlestick_writer = csv.writer(csvfile, delimiter=',')


# for candlestick in candles:
#     print(candlestick)
    
#     candlestick_writer.writerow(candlestick)

# print(len(candles))


candles = client.get_historical_klines("BTCUSDT", Client.KLINE_INTERVAL_1HOUR, "1 Jul, 2020", "3 July, 2021")

csvfile = open('July-July_1H.csv', 'w', newline='')
candlestick_writer = csv.writer(csvfile, delimiter=',')


for candlestick in candles:
    candlestick_writer.writerow(candlestick)

csvfile.close()
print(len(candles))
