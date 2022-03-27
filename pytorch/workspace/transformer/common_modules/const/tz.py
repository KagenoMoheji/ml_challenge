import datetime

'''
[タイムゾーンの適用方法例]
now = datetime.datetime.now(TZ_JST)
dt = datetime.datetime.strptime(str_datetime, "%Y-%m-%d").astimezone(TZ_JST)
dt = datetime.datetime.fromtimestamp(time.time()).astimezone(TZ_JST) # UNIX現在時刻取得
dt = datetime.datetime.strptime(str_datetime, "%Y-%m-%d").replace(TZ_JST) # 時差計算無しでタイムゾーンをすり替え
'''

TZ_UTC = datetime.timezone.utc
TZ_JST = datetime.timezone(datetime.timedelta(hours = 9), "JST")
