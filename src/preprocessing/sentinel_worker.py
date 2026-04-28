import os
import random
import boto3
import xml.etree.ElementTree as ET
from botocore import UNSIGNED
from botocore.client import Config
from shapely.geometry import Polygon
from datetime import timedelta
from esa_snappy import ProductIO, jpy
import time
import shutil


class Sentinel1Worker:
    def __init__(self, bucket="sentinel-s1-l1c", out_dir="out_safe"):
        self.bucket = bucket
        self.out_dir = out_dir
        self.s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
        os.makedirs(self.out_dir, exist_ok=True)

    def _daterange(self, start, end):
        cur = start
        while cur <= end:
            yield cur
            cur += timedelta(days=1)

    def _read_footprint(self, manifest_key):
        try:
            xml = self.s3.get_object(
                Bucket=self.bucket,
                Key=manifest_key
            )["Body"].read()

            root = ET.fromstring(xml)
            ns = {
                "safe": "http://www.esa.int/safe/sentinel-1.0",
                "gml": "http://www.opengis.net/gml"
            }

            el = root.find(
                ".//safe:frameSet/safe:frame/safe:footPrint/gml:coordinates",
                ns
            )

            if el is None:
                print(f"[WARN] footprint отсутствует: {manifest_key}")
                return None

            coords = [
                tuple(map(float, p.split(",")))
                for p in el.text.strip().split()
            ]

            return Polygon(coords)
        except Exception as e:
            print(f"[ERROR] footprint не прочитан: {manifest_key} | {e}")
            return None

    def _safe_exists(self, safe_name):
        return os.path.isdir(os.path.join(self.out_dir, safe_name))

    def _image_exists(self, safe_name, band_name):
        img = os.path.join(
            self.out_dir,
            safe_name,
            "image",
            f"{safe_name}_{band_name}.png"
        )
        return os.path.isfile(img)

    def _download_safe(self, safe_prefix):
        name = safe_prefix.rstrip("/").split("/")[-1]
        dst = os.path.join(self.out_dir, name)

        print(f"[INFO] загрузка SAFE: {name}")

        try:
            objects = []
            for page in self.s3.get_paginator("list_objects_v2").paginate(
                Bucket=self.bucket,
                Prefix=safe_prefix
            ):
                objects.extend(page.get("Contents", []))

            if not objects:
                print(f"[WARN] SAFE пустой: {name}")
                return None

            for obj in objects:
                rel = obj["Key"][len(safe_prefix):]
                path = os.path.join(dst, rel)
                os.makedirs(os.path.dirname(path), exist_ok=True)
                self.s3.download_file(self.bucket, obj["Key"], path)

        except Exception as e:
            if os.path.isdir(dst):
                shutil.rmtree(dst)
            print(f"[ERROR] ошибка загрузки SAFE {name}: {e}")
            return None

        print(f"[OK] SAFE загружен: {name}")
        return dst

    def convert_to_image(self, safe_path, band_name="Amplitude_HH", out_dir=None, format="PNG"):
        Color = jpy.get_type("java.awt.Color")
        ColorPoint = jpy.get_type("org.esa.snap.core.datamodel.ColorPaletteDef$Point")
        ColorPaletteDef = jpy.get_type("org.esa.snap.core.datamodel.ColorPaletteDef")
        ImageInfo = jpy.get_type("org.esa.snap.core.datamodel.ImageInfo")
        ImageManager = jpy.get_type("org.esa.snap.core.image.ImageManager")
        JAI = jpy.get_type("javax.media.jai.JAI")

        if out_dir is None:
            out_dir = os.path.join(safe_path, "image")
        os.makedirs(out_dir, exist_ok=True)

        name = os.path.basename(safe_path.rstrip("/"))
        out_file = os.path.join(out_dir, f"{name}_{band_name}.png")

        print(f"[INFO] обработка изображения: {name}")

        try:
            product = ProductIO.readProduct(
                os.path.join(safe_path, "manifest.safe")
            )

            band = product.getBand(band_name)
            if band is None:
                print(f"[WARN] бэнд отсутствует: {name} | {band_name}")
                product.dispose()
                return None

            palette = ColorPaletteDef([
                ColorPoint(0.0, Color.BLACK),
                ColorPoint(1000.0, Color.WHITE)
            ])

            band.setImageInfo(ImageInfo(palette))

            t0 = time.time()
            img = ImageManager.getInstance().createColoredBandImage(
                [band],
                band.getImageInfo(),
                0
            )
            JAI.create("filestore", img, out_file, format)
            product.dispose()

            print(f"[OK] изображение создано: {out_file} ({time.time() - t0:.2f} сек)")
            return out_file

        except Exception as e:
            print(f"[ERROR] ошибка обработки SNAP: {name} | {e}")
            return None

    def download_and_process_by_areas(
        self,
        start_date,
        end_date,
        areas,
        product="GRD",
        mode="EW",
        pol="DH",
        band_name="Amplitude_HH",
        max_scenes=1000,
        max_attempts=200000
    ):
        dates = list(self._daterange(start_date, end_date))
        processed = 0
        attempts = 0

        print(f"[START] диапазон {start_date} – {end_date}, лимит {max_scenes}")

        while processed < max_scenes and attempts < max_attempts:
            attempts += 1

            d = random.choice(dates)
            prefix = f"{product}/{d.year}/{d.month}/{d.day}/{mode}/{pol}/"

            resp = self.s3.list_objects_v2(
                Bucket=self.bucket,
                Prefix=prefix,
                Delimiter="/"
            )

            prefixes = resp.get("CommonPrefixes", [])
            if not prefixes:
                continue

            cp = random.choice(prefixes)
            safe_prefix = cp["Prefix"]
            safe_name = safe_prefix.rstrip("/").split("/")[-1]

            if self._image_exists(safe_name, band_name):
                print(f"[SKIP] изображение уже существует: {safe_name}")
                continue

            manifest_key = safe_prefix + "manifest.safe"
            poly = self._read_footprint(manifest_key)
            if poly is None:
                continue

            if not any(poly.intersects(a) for a in areas.values()):
                continue

            if not self._safe_exists(safe_name):
                safe_path = self._download_safe(safe_prefix)
                if not safe_path:
                    continue
            else:
                safe_path = os.path.join(self.out_dir, safe_name)
                print(f"[INFO] SAFE уже существует: {safe_name}")

            if self._image_exists(safe_name, band_name):
                print(f"[SKIP] изображение появилось ранее: {safe_name}")
                continue

            res = self.convert_to_image(
                safe_path,
                band_name=band_name
            )

            if res is None:
                print(f"[WARN] изображение не создано: {safe_name}")
                continue

            processed += 1
            print(f"[DONE] обработано {processed}/{max_scenes}: {safe_name}")

        print(f"[END] завершено, обработано {processed}, попыток {attempts}")
        return processed