# Generated by Django 4.0.4 on 2023-01-12 14:04

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('accounts', '0003_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='regform',
            name='movies',
            field=models.TextField(default=''),
        ),
    ]