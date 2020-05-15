from django.views.generic import TemplateView

class IndexView(TemplateView):
    template_name = "main/index.html"

class PriceView(TemplateView):
    template_name = "main/price.html"
