import polars as pl


def add_missing_dates(df: pl.DataFrame) -> pl.DataFrame:
    """Adds missing dates to a dataframe in long format.
    Every day has an entry for each (store_nbr, family) pair."""

    return (
        pl.DataFrame()
        .with_columns(
            pl.date_range(
                df.select(pl.col("date").min()).item(),
                df.select(pl.col("date").max()).item(),
            ).alias("date")
        )
        .join(df.select("store_nbr", "family").unique(maintain_order=True), how="cross")
        .join(df, on=["date", "store_nbr", "family"], how="full", coalesce=True)
        .fill_null(0)  # missing dates = store closed
    )


def add_series_id(df: pl.DataFrame) -> pl.DataFrame:
    """Adds a series identifier column; useful for Chronos.
    Each series is characterized by a (store_nbr, family) pair."""

    return (
        df.select("store_nbr", "family")
        .unique(maintain_order=True)
        .with_row_index("series_id")
        .join(df, on=["store_nbr", "family"])
    )


def add_store_info(df: pl.DataFrame) -> pl.DataFrame:
    """Adds store information to the dataframe."""

    return df.join(pl.read_csv("data/stores.csv"), on="store_nbr").rename(
        {"type": "store_type", "cluster": "store_cluster"}
    )


def add_holiday_events(df: pl.DataFrame) -> pl.DataFrame:
    """Adds a boolean column to the dataframe indicating holidays events."""

    events = (
        pl.read_csv("data/holidays_events.csv", try_parse_dates=True)
        .filter(transferred=False)  # keep only observed holidays
        .drop("transferred", "description", "type")
        .unique()  # drop multiple holidays in same locale
    )

    df = df.join(
        events.filter(locale="National").select("date", nat_event=True),
        on="date",
        how="left",
    )

    df = df.join(
        events.filter(locale="Regional").select("date", "locale_name", reg_event=True),
        left_on=["date", "state"],
        right_on=["date", "locale_name"],
        how="left",
    )

    df = df.join(
        events.filter(locale="Local").select("date", "locale_name", loc_event=True),
        left_on=["date", "city"],
        right_on=["date", "locale_name"],
        how="left",
    )

    return df.with_columns(
        (pl.col("nat_event") | pl.col("reg_event") | pl.col("loc_event"))
        .fill_null(False)
        .alias("event")
    ).drop("nat_event", "reg_event", "loc_event")


def make_prophet_events() -> pl.DataFrame:
    """Filter holiday events and compute the upper and lower windows.
    The result is in a format suitable for Prophet."""

    events = pl.read_csv("data/holidays_events.csv", try_parse_dates=True)

    return (
        events.rename({"description": "event"})
        .filter((pl.col("type") != "Work Day"))  # not a holiday
        .filter(~pl.col("transferred"))  # not a holiday
        .drop("type", "transferred")
        .with_columns(pl.col("event").str.strip_prefix("Traslado "))  # transferred
        .with_columns(
            pl.col("event")
            .str.split("+")
            .list.to_struct(fields=["event", "upper_window"])
        )
        .unnest("event")  # add upper_window column
        .with_columns(
            pl.col("event")
            .str.split("-")
            .list.to_struct(fields=["event", "lower_window"])
        )
        .unnest("event")  # add lower_window column
        .with_columns(  # cases when +/- does not indicate extra days are null
            -pl.col("lower_window").str.to_integer(strict=False),
            pl.col("upper_window").str.to_integer(strict=False),
        )
        .fill_null(0)  # nulls are marked as no window
        .with_columns(ad=pl.col("lower_window") == pl.col("upper_window"))  # actual day
        .with_columns(  # take window value for each holiday
            pl.col("lower_window").min().over("event"),
            pl.col("upper_window").max().over("event"),
        )
        .filter(pl.col("ad"))  # keep only the actual day with window values
        .drop("ad")
    )


def get_store_holidays(
    df: pl.DataFrame, prophet_events: pl.DataFrame, store_nbr: int
) -> pl.DataFrame:
    store = df.select("store_nbr", "city", "state").filter(store_nbr=store_nbr).unique()

    city = store["city"].item()
    state = store["state"].item()

    store_events = prophet_events.filter(
        (pl.col("locale") == "National")
        | ((pl.col("locale") == "Regional") & (pl.col("locale_name") == state))
        | ((pl.col("locale") == "Local") & (pl.col("locale_name") == city))
    )

    # format data for Prophet
    return store_events.select(
        pl.col("date").alias("ds"),
        pl.col("event").alias("holiday"),
        pl.col("lower_window"),
        pl.col("upper_window"),
    )


if __name__ == "__main__":
    train = pl.read_csv("data/train.csv", try_parse_dates=True)
    test = pl.read_csv("data/test.csv", try_parse_dates=True)

    # write separate events dataframe formatted for Prophet
    make_prophet_events().write_csv("data/preproc/prophet_events.csv")

    # some dates are missing from the original dataframes (multiple Dec 25)
    train = add_missing_dates(train)

    # work on train and test simultaneously to add features
    df = pl.concat([train, test.with_columns(sales=None).select(train.columns)])

    df = add_store_info(df)
    df = add_series_id(df)
    df = add_holiday_events(df)

    # split train and test
    train = df.join(train, on="date", how="semi")
    test = df.join(test, on="date", how="semi").drop("sales")

    train.write_csv("data/preproc/train.csv")
    test.write_csv("data/preproc/test.csv")
