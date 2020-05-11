from django.db import models


class BaseRecord(models.Model):
    name = models.CharField(max_length=200, unique=True,
                            help_text="The unique name of this entry")
    description = models.TextField(blank=True, null=True,
                                   help_text="A verbose description of this entry")
    created_at = models.DateTimeField(auto_now_add=True, db_index=True,
                                      help_text="The date and time this entry was created at")
    updated_at = models.DateTimeField(auto_now=True,
                                      help_text="The date and time this entry was last updated at")
    removed = models.BooleanField(default=False, db_index=True)

    def get_model_name(self):
        return self.__class__.__name__.lower()

    class Meta:
        abstract = True
        ordering = ['name', ]



