import sqlite3
from datetime import date, datetime, timedelta
from scripts.utils import create_directory_if_not_exist, download_file
import os

db_file = '../../asx_gym/asx_gym/db.sqlite3'

stock_index_codes = {
    'ASX20': 'xtl',
    'ASX50': 'xfl',
    'ASX100': 'xto',
    'ASX200': 'xjo',
    'ASX300': 'xko',
    'ALL ORD': 'xao',
}


def insert_stock_price_history(conn, line):
    try:
        values = line.split(',')
        code = values[0].strip()
        price_date = values[1].strip()
        price_open = values[2].strip()
        price_close = values[3].strip()
        price_high = values[4].strip()
        price_low = values[5].strip()
        stock_volume = values[6].strip()
        name = f'{code}:{price_date}'

        cur = conn.cursor()
        cur.execute(f'SELECT count(*) FROM stock_stockpricedailyhistory WHERE name="{name}"')
        row = cur.fetchone()[0]
        if row == 0:
            company_code = f'ASX:{code.upper()}'
            cur.execute(f'SELECT id FROM stock_company WHERE code="{company_code}"')
            company = cur.fetchone()
            if company:
                company_id = company[0]
                sql = '''
                    INSERT INTO 
                    stock_stockpricedailyhistory(name,price_date,open_price,
                    close_price,high_price,low_price,volume,company_id,created_at,updated_at,removed)
                    VALUES(?,?,?,?,?,?,?,?,?,?,?)
                '''
                cur.execute(sql,
                            (name, price_date, price_open, price_close,
                             price_high, price_low, stock_volume, company_id,
                             datetime.now(), datetime.now(), False))
                print(f'{code}-{price_date}-{price_open}-{price_close}-{price_high}-{price_low}-{stock_volume} created')
                conn.commit()
        else:
            print(f'{code}-{price_date}-{price_open}-{price_close}-{price_high}-{price_low}-{stock_volume} exists')
    except Exception as e:
        print(f'{line}-{str(e)}')


def insert_stock_index_history(conn, line):
    try:
        values = line.split(',')
        code = values[0].strip()
        index_date = values[1].strip()
        index_open = values[2].strip()
        index_close = values[3].strip()
        index_high = values[4].strip()
        index_low = values[5].strip()
        name = f'{stock_index_codes[code]}:{index_date}'

        cur = conn.cursor()
        cur.execute(f'SELECT count(*) FROM stock_asxindexdailyhistory WHERE name="{name}"')
        row = cur.fetchone()[0]
        if row == 0:
            sql = '''
                        INSERT INTO 
                        stock_asxindexdailyhistory(name,index_name,index_date,open_index,
                        close_index,high_index,low_index,created_at,updated_at,removed)
                        VALUES(?,?,?,?,?,?,?,?,?,?,?,?)
                    '''
            cur.execute(sql,
                        (name, code, index_date, index_open, index_close,
                         index_high, index_low,
                         datetime.now(), datetime.now(), False))
            print(f'{code}-{index_date}-{index_open}-{index_close}-{index_high}-{index_low} created')
            conn.commit()

        else:
            print(f'{code}-{index_date}-{index_open}-{index_close}-{index_high}-{index_low} exists')

    except Exception as e:
        print(f'{line}-{str(e)}')


conn = sqlite3.connect(db_file)

cur = conn.cursor()
cur.execute("SELECT data_name,updated_date FROM stock_dataupdatehistory")
rows = cur.fetchall()

for row in rows:
    data_name = row[0]
    update_date_str = row[1]
    update_date = datetime.strptime(update_date_str, '%Y-%m-%d').date()
    today = date.today()
    days = (today - update_date).days

    for day in range(-2, days - 1):
        retrieve_date = today + timedelta(days=day)
        retrieve_date_str = retrieve_date.strftime('%Y-%m-%d')

        dates = retrieve_date_str.split('-')
        year = dates[0].zfill(2)
        month = dates[1].zfill(2)
        day = dates[2].zfill(2)

        file_name = f'{data_name}/{year}/{month}/stock_{data_name}_{year}_{month}_{day}.csv'
        print(f'Downloading {file_name}')
        create_directory_if_not_exist(f'data/{data_name}/{year}/{month}')
        download_file(file_name)

        data_file = open(f'data/{file_name}')
        line = data_file.readline().strip()
        if line == '':
            os.remove(f'data/{file_name}')
        else:

            while line:
                if data_name == 'index':
                    insert_stock_index_history(conn, line)
                else:
                    insert_stock_price_history(conn, line)
                line = data_file.readline()

        data_file.close()

conn.close()
