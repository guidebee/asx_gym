from django.contrib import admin

# Register your models here.
from stock.models import Company, Sector, StockPriceDailyHistory, AsxIndexDailyHistory


@admin.register(Sector)
class SectorAdmin(admin.ModelAdmin):
    list_display = ('full_name', 'sector_type', 'number_of_companies')


@admin.register(Company)
class CompanyAdmin(admin.ModelAdmin):
    list_display = ('code', 'sector')


@admin.register(AsxIndexDailyHistory)
class AsxIndexDailyHistoryAdmin(admin.ModelAdmin):
    list_display = ('index_name', 'index_date', 'open_index', 'close_index',
                    'high_index', 'low_index')
    ordering = ('index_name', '-index_date')


@admin.register(StockPriceDailyHistory)
class StockPriceDailyHistoryAdmin(admin.ModelAdmin):
    list_display = ('company', 'price_date', 'open_price', 'close_price',
                    'high_price', 'low_price')
    ordering = ('company', '-price_date')
