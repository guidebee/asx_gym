from scripts.utils import create_directory_if_not_exist, download_file
import json
import sqlite3

db_file = '../../asx_gym/asx_gym/db.sqlite3'

create_directory_if_not_exist('data/company')
company_file = 'company/companies.json'
download_file(company_file)
sector_file = 'company/sectors.json'
download_file(sector_file)

conn = sqlite3.connect(db_file)

cur = conn.cursor()

with open('data/company/sectors.json') as f:
    sectors = json.load(f)
    for sector in sectors:
        pk = sector['pk']
        cur.execute(f'SELECT count(*) FROM stock_sector WHERE id={pk}')
        row = cur.fetchone()[0]
        full_name = sector['fields']['full_name']
        if row == 0:
            name = sector['fields']['name']
            parent_sector = sector['fields']['parent_sector']
            created_at = sector['fields']['created_at']
            updated_at = sector['fields']['updated_at']
            removed = sector['fields']['removed']
            sector_type = sector['fields']['sector_type']
            sector_id = sector['fields']['sector_id']
            sector_index = sector['fields']['sector_index']
            number_of_companies = sector['fields']['number_of_companies']
            lft = sector['fields']['lft']
            rght = sector['fields']['rght']
            tree_id = sector['fields']['tree_id']
            sector_level = sector['fields']['sector_level']
            sql = '''
                INSERT INTO stock_sector(id,name,full_name,created_at,updated_at,
                removed,lft,rght,tree_id,sector_level,number_of_companies,
                sector_id,sector_index,sector_type,parent_sector_id)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            '''
            cur.execute(sql, (pk, name, full_name, created_at, updated_at,
                              removed, lft, rght, tree_id, sector_level, number_of_companies,
                              sector_id, sector_index, sector_type, parent_sector))
            conn.commit()
            print(f'sector {full_name} created')
        else:
            print(f'sector {full_name} exists')

with open('data/company/companies.json') as f:
    companies = json.load(f)
    for company in companies:
        pk = company['pk']
        cur.execute(f'SELECT count(*) FROM stock_company WHERE id={pk}')
        row = cur.fetchone()[0]
        name = company['fields']['name']
        if row == 0:
            description = company['fields']['description']
            created_at = company['fields']['created_at']
            updated_at = company['fields']['updated_at']
            removed = company['fields']['removed']
            sector = company['fields']['sector']
            code = company['fields']['code']
            market_capacity = company['fields']['market_capacity']
            sql = '''
             INSERT INTO stock_company(id,name,description,created_at,updated_at,
             removed,code,market_capacity,sector_id) 
             VALUES(?,?,?,?,?,?,?,?,?)
            '''
            cur.execute(sql, (pk, name, description, created_at, updated_at,
                              removed, code, market_capacity, sector))

            conn.commit()
            print(f'company {name} created')
        else:
            print(f'company {name} exists')

conn.close()
