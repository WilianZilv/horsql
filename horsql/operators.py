from common import is_iterable


class Operator:
    def __init__(self, value, operator):
        self.value = value
        self.operator = operator

    def build(self):
        return f"{self.operator} %s"

    def params(self):
        return [self.value]


class Equals(Operator):
    def __init__(self, value):
        super().__init__(value, "=")


class NotEquals(Operator):
    def __init__(self, value):
        super().__init__(value, "<>")


class GreaterThan(Operator):
    def __init__(self, value):
        super().__init__(value, ">")


class GreaterOrEqualsThan(Operator):
    def __init__(self, value):
        super().__init__(value, ">=")


class EqualsOrGreaterThan(Operator):
    def __init__(self, value):
        super().__init__(value, ">=")


class LessThan(Operator):
    def __init__(self, value):
        super().__init__(value, "<")


class LessOrEqualsThan(Operator):
    def __init__(self, value):
        super().__init__(value, "<=")


class EqualsOrLessThan(Operator):
    def __init__(self, value):
        super().__init__(value, "<=")


class IsIn(Operator):
    def __init__(self, value):
        super().__init__(value, "in")


class NotIn(Operator):
    def __init__(self, value):
        super().__init__(value, "not in")


class Like(Operator):
    def __init__(self, value):
        super().__init__(value, "like")


class Ilike(Operator):
    def __init__(self, value):
        super().__init__(value, "ilike")


class NotLike(Operator):
    def __init__(self, value):
        super().__init__(value, "not like")


class NotIlike(Operator):
    def __init__(self, value):
        super().__init__(value, "not ilike")


class Between(Operator):
    def __init__(self, value):
        self.value = value
        self.operator = "between"

    def build(self):
        return f"{self.operator} %s and %s"

    def params(self):
        return self.value


class Not(Operator):
    def __init__(self, value):
        if is_iterable(value):
            self.custom = NotIn(value)
        else:
            self.custom = NotEquals(value)

    def build(self):
        return self.custom.build()

    def params(self):
        return self.custom.params()


class Is(Operator):
    def __init__(self, value):
        if is_iterable(value):
            self.custom = IsIn(value)
        else:
            self.custom = Equals(value)

    def build(self):
        return self.custom.build()

    def params(self):
        return self.custom.params()


class IsNull(Operator):
    def __init__(self):
        self.operator = "is null"

    def build(self):
        return f"{self.operator}"

    def params(self):
        return []


class NotNull(Operator):
    def __init__(self):
        self.operator = "notnull"

    def build(self):
        return f"{self.operator}"

    def params(self):
        return []


class And(dict):
    condition = "and"


class Or(dict):
    condition = "or"


class Column:
    token = "@"

    def __init__(self, value):
        self.value = value

    def build(self):
        return f"{self.token}{self.value}"

    @staticmethod
    def destroy(query: str):
        parts = query.split(" ")

        token = Column.token

        for i, part in enumerate(parts):
            if part.startswith(f"'{token}"):
                parts[i] = part.strip()[2:-1]

        return " ".join(parts)
