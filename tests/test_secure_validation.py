"""
Test script to verify comprehensive server-side validation and secure error handling
"""
import requests
import json
from datetime import datetime, timedelta


BASE_URL = "http://localhost:10242"


def get_token(username="testuser123", password="SecurePass123!"):
    response = requests.post(f"{BASE_URL}/token", data={"username": username, "password": password})
    if response.status_code == 200 and "access_token" in response.json():
        return response.json()["access_token"]
    return None


def get_auth_headers(username="testuser123", password="SecurePass123!"):
    token = get_token(username, password)
    return {"Authorization": f"Bearer {token}"} if token else {}


def test_validation_errors():
    """Test that validation errors don't leak sensitive information"""
    print("=== Testing Validation Errors ===")
    
    # Test 1: Invalid user creation with malicious input
    print("\n1. Testing user creation with malicious input...")
    malicious_data = {
        "name": "<script>alert('xss')</script>",
        "username": "'; DROP TABLE users; --",
        "password": "weak",
        "is_admin": "not_a_boolean"
    }
    try:
        response = requests.post(f"{BASE_URL}/users", json=malicious_data)
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"Exception during user creation: {e}")
    
    # Test 2: SQL injection attempt in login
    print("\n2. Testing SQL injection in login...")
    login_data = {
        "username": "admin'; DROP TABLE users; --",
        "password": "password123"
    }
    
    response = requests.post(f"{BASE_URL}/token", data=login_data)
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    # Test 3: Invalid data types
    print("\n3. Testing invalid data types...")
    invalid_activity = {
        "title": 123,  # Should be string
        "start_date": "not_a_date",
        "time": "25:99",  # Invalid time
        "category_id": "not_an_integer",
        "mode": "invalid_mode"
    }
    
    response = requests.post(f"{BASE_URL}/activities", json=invalid_activity)
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")


def test_path_parameter_validation():
    """Test that path parameters are properly validated"""
    print("\n=== Testing Path Parameter Validation ===")
    
    # Test 1: Negative ID
    print("\n1. Testing negative user ID...")
    response = requests.get(f"{BASE_URL}/users/-1")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    # Test 2: Zero ID
    print("\n2. Testing zero category ID...")
    response = requests.get(f"{BASE_URL}/categories/0")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    # Test 3: Non-existent resource
    print("\n3. Testing non-existent activity...")
    response = requests.get(f"{BASE_URL}/activities/99999")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")


def test_business_logic_validation():
    """Test business logic validation"""
    print("\n=== Testing Business Logic Validation ===")
    
    # First create a valid user for testing
    user_data = {
        "name": "Test User",
        "username": "testuser123",
        "password": "SecurePass123!",
        "is_admin": False
    }
    
    user_response = requests.post(f"{BASE_URL}/users", json=user_data)
    if user_response.status_code == 201 or user_response.status_code == 200:
        print("Test user created successfully")
        user_id = user_response.json().get("id")
    else:
        print("Failed to create test user, using ID 1 for testing")
        user_id = 1
    
    # Test 1: Duplicate username
    print("\n1. Testing duplicate username...")
    duplicate_user = {
        "name": "Another User",
        "username": "testuser123",  # Same username
        "password": "AnotherPass123!",
        "is_admin": False
    }
    
    response = requests.post(f"{BASE_URL}/users", json=duplicate_user)
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    # Test 2: Password change with wrong old password
    print("\n2. Testing password change with wrong old password...")
    password_data = {
        "old_password": "WrongPassword123!",
        "new_password": "NewSecurePass123!"
    }
    headers = get_auth_headers()
    response = requests.post(f"{BASE_URL}/users/{user_id}/change-password", json=password_data, headers=headers)
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")


def test_input_sanitization():
    """Test that input is properly sanitized"""
    print("\n=== Testing Input Sanitization ===")
    # Ensure test user exists for authentication
    user_data = {
        "name": "Test User",
        "username": "testuser123",
        "password": "SecurePass123!",
        "is_admin": False
    }
    user_response = requests.post(f"{BASE_URL}/users", json=user_data)
    # Test 1: HTML/JavaScript injection in category name
    print("\n1. Testing HTML injection in category creation...")
    malicious_category = {
        "name": "<img src=x onerror=alert('XSS')>",
        "icon_name": "test-icon",
        "mode": "personal"
    }
    headers = get_auth_headers()
    response = requests.post(f"{BASE_URL}/categories", json=malicious_category, headers=headers)
    print(f"Status: {response.status_code}")
    if response.status_code == 200 or response.status_code == 201:
        category = response.json()
        print(f"Sanitized name: {category.get('name')}")
        # Clean up
        requests.delete(f"{BASE_URL}/categories/{category.get('id')}", headers=headers)
    else:
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    # Test 2: Long input strings
    print("\n2. Testing extremely long input...")
    long_string = "A" * 1000  # 1000 characters
    long_input_data = {
        "name": long_string,
        "username": "test",
        "password": "ValidPass123!",
        "is_admin": False
    }
    
    response = requests.post(f"{BASE_URL}/users", json=long_input_data)
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")


def test_database_error_handling():
    """Test database error handling"""
    print("\n=== Testing Database Error Handling ===")
    
    # Test 1: Foreign key constraint violation
    print("\n1. Testing foreign key constraint...")
    invalid_activity = {
        "title": "Test Activity",
        "start_date": (datetime.now() + timedelta(days=1)).isoformat(),
        "time": "10:00",
        "category_id": 99999,  # Non-existent category
        "mode": "personal",
        "responsible_ids": [],
        "todos": []
    }
    
    response = requests.post(f"{BASE_URL}/activities", json=invalid_activity)
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")


def main():
    """Run all tests"""
    print("Starting comprehensive validation and error handling tests...")
    print("Make sure the FastAPI server is running on localhost:10242")
    
    try:
        # Test server connectivity
        response = requests.get(f"{BASE_URL}/users")
        print(f"Server connectivity test: {response.status_code}")
        
        test_validation_errors()
        test_path_parameter_validation()
        test_business_logic_validation()
        test_input_sanitization()
        test_database_error_handling()
        
        print("\n=== Test Summary ===")
        print("✅ All tests completed!")
        print("📝 Check the server logs (app_errors.log) for detailed error information")
        print("🔒 Client responses should not contain sensitive information")
        
    except requests.ConnectionError:
        print("❌ Could not connect to server. Make sure FastAPI is running on localhost:10242")
    except Exception as e:
        print(f"❌ Test error: {e}")


if __name__ == "__main__":
    main()
