# Generated by Django 4.1 on 2024-03-19 06:16

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("submit", "0003_rename_perdict_batch_size_task_predict_batch_size"),
    ]

    operations = [
        migrations.AddField(
            model_name="trainparameters",
            name="dataset",
            field=models.FileField(
                max_length=128, null=True, upload_to="dataset/", verbose_name="dataset"
            ),
        ),
    ]
