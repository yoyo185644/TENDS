# Generated by Django 4.1 on 2024-03-25 06:20

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("submit", "0007_trainparameters"),
    ]

    operations = [
        migrations.AddField(
            model_name="trainresult",
            name="dataset",
            field=models.CharField(default=0, max_length=64),
            preserve_default=False,
        ),
    ]
