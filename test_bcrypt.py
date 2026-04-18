from passlib.context import CryptContext

pwd = CryptContext(schemes=["bcrypt"], deprecated="auto")

hash = pwd.hash("admin123")
print("HASH:", hash)

print("VERIFY:", pwd.verify("admin123", hash))