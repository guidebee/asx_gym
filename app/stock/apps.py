from django.apps import AppConfig
import logging

logger = logging.getLogger(__name__)


class StockConfig(AppConfig):
    name = 'stock'

    def ready(self):
        from background_task.models import Task
        task_update_daily_stock_price = Task.objects.filter(
            task_name='stock.tasks.cron_update_daily_stock_price').first()
        if not task_update_daily_stock_price:
            from stock.tasks import cron_update_daily_stock_price
            cron_update_daily_stock_price(repeat=Task.DAILY)
            logger.info("start cron_update_daily_stock_price task")

        task_update_live_stock_price = Task.objects.filter(
            task_name='stock.tasks.cron_update_live_stock_price').first()
        if not task_update_live_stock_price:
            from stock.tasks import cron_update_live_stock_price
            cron_update_live_stock_price(repeat=60 * 20)
            logger.info("start cron_update_live_stock_price task")
