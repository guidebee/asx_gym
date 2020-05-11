from datetime import datetime

from background_task import background

from stock.management.commands.utils import update_all_stock_info, \
    get_live_stock_price, stock_indexes, get_live_stock_index
from stock.models import Company
import logging

logger = logging.getLogger(__name__)


def need_update():
    t = datetime.now()
    weekday = t.weekday()
    hour = t.hour
    return (weekday >= 0) and (weekday < 5) and (hour >= 5) and (hour <= 17)


@background(schedule=3600)
def cron_update_daily_stock_price():
    if need_update():
        try:
            logger.info(f'cron_update_daily_stock_price run')
            update_all_stock_info()
        except Exception as e:
            logger.warning(f'cron_update_daily_stock_price failed with {str(e)}')
    else:
        logger.info('bypass cron_update_daily_stock_price run')


@background(schedule=30)
def cron_update_live_stock_price():
    if need_update():
        try:
            logger.info(f'cron_update_live_stock_price run')
            companies = Company.objects.all()
            for company in companies:
                get_live_stock_price(company)
            for stock_index_code in stock_indexes:
                get_live_stock_index(stock_index_code)
        except Exception as e:
            logger.warning(f'cron_update_live_stock_price failed with {str(e)}')
    else:
        logger.info('bypass cron_update_live_stock_price run')
