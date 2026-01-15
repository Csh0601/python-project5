from django.db import models
from django.contrib.auth.models import User


class SearchRecord(models.Model):
    """用户搜索记录模型"""
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='search_records')
    query_image = models.ImageField(upload_to='queries/')
    results = models.JSONField(default=list)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.user.username} - {self.created_at.strftime('%Y-%m-%d %H:%M')}"
