import chronos
import polars as pl

train_df = pl.read_csv("data/preproc/train.csv", try_parse_dates=True)
train_df = train_df.drop("id", "city", "state", "store_type", "store_cluster")

test_df = pl.read_csv("data/preproc/test.csv", try_parse_dates=True)
test_df = test_df.drop("id", "city", "state", "store_type", "store_cluster")

pipeline = chronos.Chronos2Pipeline.from_pretrained("amazon/chronos-2")

predt_df = pl.from_pandas(
    pipeline.predict_df(
        train_df.to_pandas(),
        future_df=test_df.to_pandas(),
        prediction_length=16,
        quantile_levels=[0.5],
        id_column="series_id",
        timestamp_column="date",
        target="sales",
    )
).cast({"date": pl.Date})


sub_df = (
    pl.read_csv("data/preproc/test.csv", try_parse_dates=True)
    .select("date", "series_id", "id")
    .join(predt_df, on=["date", "series_id"])
    .select("id", pl.col("predictions").alias("sales"))
    .sort("id")
)

sub_df.write_csv("data/submission.csv")
