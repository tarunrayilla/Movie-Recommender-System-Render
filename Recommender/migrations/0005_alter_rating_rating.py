# Generated by Django 4.0.4 on 2023-01-02 11:02

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('Recommender', '0004_alter_newuser_userid'),
    ]

    operations = [
        migrations.AlterField(
            model_name='rating',
            name='rating',
            field=models.IntegerField(choices=[(5, '5'), (4, '4'), (3, '3'), (2, '2'), (1, '1')], default=0),
        ),
    ]