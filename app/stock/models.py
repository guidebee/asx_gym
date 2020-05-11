from django.db import models

# Create your models here.
from mptt.models import MPTTModel, TreeForeignKey

from base.models import BaseRecord


class Sector(MPTTModel):
    name = models.CharField('Name', max_length=128, db_index=True, )
    full_name = models.TextField('Fullname', help_text='Name of sector ', blank=True)
    parent_sector = TreeForeignKey('self', on_delete=models.CASCADE, null=True, blank=True, related_name='sub_sectors')
    created_at = models.DateTimeField('Created', db_index=True, auto_now_add=True)
    updated_at = models.DateTimeField('Last Modified', db_index=True, auto_now=True)
    removed = models.BooleanField(default=False, db_index=True)
    sector_type = models.CharField('Type', max_length=128, null=True)
    sector_id = models.IntegerField('internal id', unique=True, null=True)
    sector_index = models.CharField('Sector Index', max_length=128, null=True)
    number_of_companies = models.IntegerField('Number of companies', default=0)

    class MPTTMeta:
        order_insertion_by = ['name']
        parent_attr = 'parent_sector'
        level_attr = 'sector_level'

    class Meta:
        verbose_name_plural = 'Sectors'
        app_label = "stock"

    def __str__(self):
        return '{}-{}'.format(self.id, self.name)

    def get_all_parents(self, list_in):
        if self not in list_in:
            list_in.append(self)

        if self.parent_sector:
            self.parent_sector.get_all_parents(list_in)

    def get_all_children(self, list_in):
        if self not in list_in:
            list_in.append(self)

        if self.sub_sectors:
            for sub_sector in self.sub_sectors.all():
                sub_sector.get_all_children(list_in)

    def save(self, *args, **kwargs):
        if self.parent_sector:
            self.full_name = self.parent_sector.full_name + '/' + self.name
        else:
            self.full_name = self.name
        super(Sector, self).save(*args, **kwargs)


class Company(BaseRecord):
    sector = models.ForeignKey(Sector, on_delete=models.SET_NULL, null=True, related_name="companies")

    code = models.CharField(max_length=16, unique=True,
                            help_text="The unique code of this company")

    market_capacity = models.DecimalField(max_digits=19, decimal_places=3, help_text="Market capacity")

    def __str__(self):
        return '{}-{}'.format(self.id, self.name)

    class Meta:
        app_label = "stock"


class StockPriceDailyHistory(BaseRecord):
    company = models.ForeignKey(Company, on_delete=models.CASCADE, null=True, related_name="stock_price_daily_history")
    price_date = models.DateField()
    open_price = models.DecimalField(max_digits=10, decimal_places=3, default=0)
    close_price = models.DecimalField(max_digits=10, decimal_places=3, default=0)
    high_price = models.DecimalField(max_digits=10, decimal_places=3, default=0)
    low_price = models.DecimalField(max_digits=10, decimal_places=3, default=0)
    volume = models.DecimalField(max_digits=19, decimal_places=3, default=0)

    def __str__(self):
        return '{}-{}'.format(self.id, self.name)

    class Meta:
        app_label = "stock"


class StockPriceHistory(BaseRecord):
    company = models.ForeignKey(Company, on_delete=models.CASCADE, null=True, related_name="stock_price_history")
    price_date = models.DateTimeField()
    open_price = models.DecimalField(max_digits=10, decimal_places=3, default=0)
    close_price = models.DecimalField(max_digits=10, decimal_places=3, default=0)
    high_price = models.DecimalField(max_digits=10, decimal_places=3, default=0)
    low_price = models.DecimalField(max_digits=10, decimal_places=3, default=0)
    volume = models.DecimalField(max_digits=19, decimal_places=3, default=0)
    ask_price = models.DecimalField(max_digits=10, decimal_places=3, default=0)
    bid_price = models.DecimalField(max_digits=10, decimal_places=3, default=0)
    price = models.DecimalField(max_digits=10, decimal_places=3, default=0)
    trade_count = models.DecimalField(max_digits=10, decimal_places=3, default=0)
    trade_value = models.DecimalField(max_digits=19, decimal_places=3, default=0)
    price_change = models.DecimalField(max_digits=10, decimal_places=3, default=0)
    price_change_percentage = models.DecimalField(max_digits=10, decimal_places=3, default=0)

    def __str__(self):
        return '{}-{}'.format(self.id, self.name)

    class Meta:
        app_label = "stock"


class AsxIndexDailyHistory(BaseRecord):
    index_name = models.CharField('Index Name', max_length=128)
    index_date = models.DateField()
    open_index = models.DecimalField(max_digits=10, decimal_places=3, default=0)
    close_index = models.DecimalField(max_digits=10, decimal_places=3, default=0)
    high_index = models.DecimalField(max_digits=10, decimal_places=3, default=0)
    low_index = models.DecimalField(max_digits=10, decimal_places=3, default=0)

    def __str__(self):
        return '{}-{}'.format(self.id, self.name)

    class Meta:
        app_label = "stock"


class AsxIndexHistory(BaseRecord):
    index_name = models.CharField('Index Name', max_length=128)
    index_date = models.DateTimeField()
    close_index = models.DecimalField(max_digits=10, decimal_places=3, default=0)

    def __str__(self):
        return '{}-{}'.format(self.id, self.name)

    class Meta:
        app_label = "stock"


class InitialDataUpdateHistory(models.Model):
    data_name = models.CharField('Data Name', max_length=128)
    code = models.CharField('Code Name', max_length=128)
    updated_date = models.DateField()

    def __str__(self):
        return '{}-{}-{}-{}'.format(self.id, self.data_name, self.code, self.updated_date)

    class Meta:
        app_label = "stock"


class DataUpdateHistory(models.Model):
    data_name = models.CharField('Data Name', max_length=128)
    updated_date = models.DateField()

    def __str__(self):
        return '{}-{}-{}'.format(self.id, self.data_name, self.updated_date)

    class Meta:
        app_label = "stock"
