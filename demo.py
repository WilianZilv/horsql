import horsql

db = horsql.connect(
    database="rps",
    host="localhost:5432",
    user="dev",
    password="dev",
    dialect="postgresql",
)

db.settings.depara_geral.get()
