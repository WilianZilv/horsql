# horsql

An experimental SQL ORM that operates with Pandas DataFrames, offering the flexibility of dynamic querying without the need for model or schema definitions.

## Install

```bash
$ pip install horsql
```

## Connect

```python
db = horsql.connect(
    database="mydb",
    host="localhost",
    port=5432,
    user="dev",
    password="dev",
    dialect="postgresql", # or "mysql"
    echo=False,
    pool_size=5, # defaults to 5
)
```

## Querying

```python
import horsql.operators as o

# Get all users
df = db.public.users.get()

# Get some columns of the users that are +18
df = db.public.users.get(columns=["user_name", "age"], age=o.EqualsOrGreaterThan(18))

# (Aggregation) get the average age by city from Brazil
df = db.public.users.get(columns=["city"], avg=["age"], country="Brazil")

# Distinct columns
df = db.public.users.get(distinct=["last_name"])

# get users that were born between those dates
df = db.public.users.get(birth_date=o.Between(["1998-01-01", "2000-01-01"]))

# get users that it's last_name is Silva or Machado
df = db.public.users.get(last_name=["Silva", "Machado"])

# get users that it's last_name is not Silva or Machado
df = db.public.users.get(last_name=o.Not(["Silva", "Machado"]))

# get users from Brazil or those that have "Silva" as last_name
df = db.public.users.get(
    where=o.Or(
        country="Brazil",
        last_name="Silva"
    )
)

# get users from United States OR those that have "Smith" as last_name
# OR (first_name is "Wilian" AND last_name is "Silva")
df = db.public.users.get(
    where=o.Or(
        country="United States",
        last_name="Smith",
        o.And(
            first_name="Wilian",
            last_name="Silva"
        )
    )
)

# Ordering
df = db.public.users.order_by("age", ascending=True).get()

df = db.public.users.order_by(["age", "country"], ascending=[True, False]).get()

# Limit
df = db.public.users.limit(limit=10).get()

df = db.public.users.order_by("age", ascending=True).limit(limit=10).get()

# Pagination
df = db.public.users.paginate(page=1, page_size=10).get()

df = db.public.users.order_by("age", ascending=True).paginate(page=1, page_size=10).get()


```

## Creating/Updating records in the database

```python
new_user = pd.DataFrame([
    {
        "user_name": "WilianZilv",
        "first_name": "Wilian",
        "last_name": "Silva"
    }
])

# Create new records based on a dataframe
db.public.users.create(new_user)

# Upsert
db.public.users.create(new_user, on_conflict=["user_name"], update=["city", "country"])

# Updating records
new_user["city"] = "Curitiba"
new_user["country"] = "Brazil"

db.public.users.update(new_user, on_conflict=["user_name"], update=["city", "country"])

# Delete records
db.public.users.delete(user_name="WilianZilv")
```

# Function definitions

```python
Columns = Union[str, List[str]]

# Functions available in the Table object

def get(
    self,
    columns: Optional[Columns] = None,
    distinct: Optional[Columns] = None,
    min: Optional[Columns] = None,
    max: Optional[Columns] = None,
    sum: Optional[Columns] = None,
    avg: Optional[Columns] = None,
    count: Optional[Columns] = None,
    where: Optional[Union[And, Or]] = None,
    **and_where,
):
    ...

def create(
    self,
    df: pd.DataFrame,
    on_conflict: Optional[Columns] = None,
    update: Optional[Columns] = None,
    commit: bool = True,
):
    ...

def update(
    self,
    df: pd.DataFrame,
    on_conflict: Columns,
    update: Columns,
    commit: bool = True,
):
    ...

def limit(self, limit: int):
    ...

def paginate(self, page: int, page_size: int):
    ...

def order_by(
    self,
    columns: Union[List[str], str],
    ascending: Union[List[bool], bool] = []
):
    ...

```
