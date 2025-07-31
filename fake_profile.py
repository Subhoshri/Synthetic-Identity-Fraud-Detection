from faker import Faker
import random

fake = Faker()

def generate_fake_profile(user_id):
    profile = fake.simple_profile()
    return {
        "user_id": user_id,
        "name": profile["name"],
        "account_number": str(fake.random_number(digits=12, fix_len=True)),
        "email": profile["mail"],
        "phone": fake.phone_number(),
        "address": fake.address().replace("\n", ", "),
        "city": fake.city(),
        "state": fake.state(),
        "zipcode": fake.zipcode(),
        "country": fake.country(),
        "account_type": random.choice(["Savings", "Current", "Salary"]),
        "account_opening_date": str(fake.date_between(start_date='-5y', end_date='-1y')),
        "last_login": str(fake.date_time_this_month()),
        "login_count": random.randint(1, 100),
        "verification_status": {
            "email": random.choice([True, False]),
            "phone": random.choice([True, False]),
            "identity": random.choice([True, False]),
            "address": random.choice([True, False]),
            "kyc": random.choice([True, False])
        },
        "device_info": {
            "device_type": random.choice(["Mobile", "Desktop", "Tablet"]),
            "device_os": random.choice(["windows", "macos", "linux", "android"]),
            "device_count": random.randint(1, 5),
            "last_ip": fake.ipv4(),
            "location": fake.city()
        }
    }
