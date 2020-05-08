from scripts.utils import create_directory_if_not_exist, download_file

create_directory_if_not_exist('data/company')
company_file = 'company/companies.json'
download_file(company_file)
sector_file = 'company/sectors.json'
download_file(sector_file)
