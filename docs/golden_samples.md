# Golden Sample Definitions: Reference Standards for Gap Analysis

## Overview

This document defines golden samples for each artifact type - idealized examples that demonstrate perfect or near-perfect documentation standards. These samples serve as reference points for rule development, testing, and validation, ensuring consistent quality expectations across the gap analysis system.

## Purpose and Usage

### Testing Reference Standards
- **Rule Validation**: Test rules against known-good samples to verify correct behavior
- **Confidence Calibration**: Establish confidence score baselines for high-quality content
- **Regression Testing**: Detect rule changes that affect scoring consistency
- **Performance Benchmarking**: Measure rule execution against standardized inputs

### Quality Benchmarking
- **Documentation Standards**: Define what "excellent" documentation looks like per artifact type
- **Team Training**: Provide examples for developers to understand documentation expectations
- **Automated Scoring**: Establish confidence score ranges for different quality levels
- **Continuous Improvement**: Track documentation quality improvements over time

---

## Kubernetes Golden Samples

### Perfect Deployment Sample
**Expected Confidence Score**: 0.95-1.0  
**Expected Rule Passes**: All documentation and security rules

```yaml
# Golden Sample: Kubernetes Deployment - Perfect Documentation
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nginx-production
  namespace: web-services
  labels:
    app: nginx
    version: "1.21.6"
    component: web-server
    environment: production
    team: platform
    cost-center: engineering
  annotations:
    description: "Production nginx web server for static content delivery and reverse proxy"
    documentation: "https://wiki.company.com/nginx-production"
    contact: "platform-team@company.com"
    deployment.kubernetes.io/revision: "3"
    config.linkerd.io/proxy-cpu-request: "100m"
    config.linkerd.io/proxy-memory-request: "20Mi"
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxUnavailable: 1
      maxSurge: 1
  selector:
    matchLabels:
      app: nginx
      version: "1.21.6"
  template:
    metadata:
      labels:
        app: nginx
        version: "1.21.6"
        component: web-server
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "9113"
        prometheus.io/path: "/metrics"
    spec:
      securityContext:
        runAsNonRoot: true
        runAsUser: 101
        fsGroup: 101
        seccompProfile:
          type: RuntimeDefault
      containers:
      - name: nginx
        image: nginx:1.21.6-alpine
        imagePullPolicy: IfNotPresent
        ports:
        - name: http
          containerPort: 8080
          protocol: TCP
        - name: metrics
          containerPort: 9113
          protocol: TCP
        env:
        - name: NGINX_PORT
          value: "8080"
        resources:
          requests:
            memory: "64Mi"
            cpu: "100m"
          limits:
            memory: "256Mi"
            cpu: "500m"
        securityContext:
          allowPrivilegeEscalation: false
          readOnlyRootFilesystem: true
          capabilities:
            drop:
            - ALL
        livenessProbe:
          httpGet:
            path: /health
            port: http
          initialDelaySeconds: 30
          periodSeconds: 10
          timeoutSeconds: 5
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /ready
            port: http
          initialDelaySeconds: 5
          periodSeconds: 5
          timeoutSeconds: 3
          failureThreshold: 3
        volumeMounts:
        - name: tmp-volume
          mountPath: /tmp
        - name: nginx-config
          mountPath: /etc/nginx/conf.d
          readOnly: true
      volumes:
      - name: tmp-volume
        emptyDir: {}
      - name: nginx-config
        configMap:
          name: nginx-config
      nodeSelector:
        kubernetes.io/arch: amd64
        node-type: web
      tolerations:
      - key: node-type
        operator: Equal
        value: web
        effect: NoSchedule
      affinity:
        podAntiAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
          - weight: 100
            podAffinityTerm:
              labelSelector:
                matchExpressions:
                - key: app
                  operator: In
                  values:
                  - nginx
              topologyKey: kubernetes.io/hostname
```

### High-Quality Service Sample
**Expected Confidence Score**: 0.85-0.95  
**Expected Rule Passes**: Most rules, minor optional documentation missing

```yaml
# Golden Sample: Kubernetes Service - High Quality
apiVersion: v1
kind: Service
metadata:
  name: nginx-service
  namespace: web-services
  labels:
    app: nginx
    component: web-server
    environment: production
  annotations:
    description: "Load balancer service for nginx production deployment"
    service.beta.kubernetes.io/aws-load-balancer-type: "nlb"
    service.beta.kubernetes.io/aws-load-balancer-cross-zone-load-balancing-enabled: "true"
spec:
  type: LoadBalancer
  ports:
  - name: http
    port: 80
    targetPort: http
    protocol: TCP
  - name: https
    port: 443
    targetPort: http
    protocol: TCP
  selector:
    app: nginx
    version: "1.21.6"
  sessionAffinity: None
  loadBalancerSourceRanges:
  - "10.0.0.0/8"
  - "172.16.0.0/12"
```

### Moderate Quality Sample
**Expected Confidence Score**: 0.60-0.75  
**Expected Rule Passes**: Basic requirements met, several optional fields missing

```yaml
# Golden Sample: Kubernetes ConfigMap - Moderate Quality
apiVersion: v1
kind: ConfigMap
metadata:
  name: nginx-config
  namespace: web-services
  labels:
    app: nginx
    component: config
data:
  default.conf: |
    server {
        listen 8080;
        server_name localhost;
        
        location / {
            root /usr/share/nginx/html;
            index index.html;
        }
        
        location /health {
            access_log off;
            return 200 "healthy\n";
            add_header Content-Type text/plain;
        }
    }
```

---

## FastAPI Golden Samples

### Perfect Endpoint Sample
**Expected Confidence Score**: 0.95-1.0  
**Expected Rule Passes**: All documentation, validation, and error handling rules

```python
# Golden Sample: FastAPI Endpoint - Perfect Documentation
from typing import Optional, List
from fastapi import FastAPI, HTTPException, Depends, status, Query, Path
from pydantic import BaseModel, Field, EmailStr
from datetime import datetime

class UserResponse(BaseModel):
    """Response model for user data with comprehensive field documentation."""
    
    id: int = Field(..., description="Unique user identifier", example=12345)
    email: EmailStr = Field(..., description="User's email address", example="user@example.com")
    name: str = Field(..., min_length=1, max_length=100, description="User's full name", example="John Doe")
    created_at: datetime = Field(..., description="Account creation timestamp", example="2023-01-15T10:30:00Z")
    is_active: bool = Field(..., description="Whether the user account is currently active", example=True)
    profile_url: Optional[str] = Field(None, description="Optional URL to user's profile page", example="https://api.example.com/users/12345")

    class Config:
        schema_extra = {
            "example": {
                "id": 12345,
                "email": "john.doe@example.com",
                "name": "John Doe",
                "created_at": "2023-01-15T10:30:00Z",
                "is_active": True,
                "profile_url": "https://api.example.com/users/12345"
            }
        }

class UserCreate(BaseModel):
    """Request model for creating new users with validation rules."""
    
    email: EmailStr = Field(..., description="Valid email address for the new user")
    name: str = Field(..., min_length=1, max_length=100, description="User's full name")
    password: str = Field(..., min_length=8, description="Password must be at least 8 characters long")

class ErrorResponse(BaseModel):
    """Standardized error response format for all API endpoints."""
    
    error: str = Field(..., description="Error type identifier", example="USER_NOT_FOUND")
    message: str = Field(..., description="Human-readable error description", example="User with ID 12345 was not found")
    details: Optional[dict] = Field(None, description="Additional error context", example={"user_id": 12345})

@app.get(
    "/users/{user_id}",
    response_model=UserResponse,
    status_code=status.HTTP_200_OK,
    summary="Retrieve user by ID",
    description="Fetch detailed information for a specific user account by their unique identifier.",
    response_description="User data including profile information and account status",
    responses={
        200: {
            "description": "User found and returned successfully",
            "model": UserResponse
        },
        404: {
            "description": "User not found",
            "model": ErrorResponse,
            "content": {
                "application/json": {
                    "example": {
                        "error": "USER_NOT_FOUND",
                        "message": "User with ID 12345 was not found",
                        "details": {"user_id": 12345}
                    }
                }
            }
        },
        422: {
            "description": "Invalid user ID format",
            "model": ErrorResponse
        },
        500: {
            "description": "Internal server error",
            "model": ErrorResponse
        }
    },
    tags=["Users"]
)
async def get_user(
    user_id: int = Path(
        ..., 
        ge=1, 
        description="Unique user identifier (positive integer)",
        example=12345
    ),
    include_profile: bool = Query(
        False,
        description="Whether to include extended profile information in the response",
        example=True
    ),
    current_user = Depends(get_current_active_user)
) -> UserResponse:
    """
    Retrieve comprehensive user information by unique identifier.
    
    This endpoint fetches detailed user account information including profile data,
    account status, and creation metadata. Access requires valid authentication.
    
    **Authentication Required**: Valid JWT token in Authorization header
    
    **Rate Limiting**: 100 requests per minute per user
    
    **Caching**: Response cached for 5 minutes for non-profile requests
    
    Args:
        user_id: The unique integer identifier for the user account (must be positive)
        include_profile: Optional flag to include extended profile information
        current_user: Authenticated user context (injected by dependency)
    
    Returns:
        UserResponse: Complete user account information
        
    Raises:
        HTTPException: 
            - 404: User not found in database
            - 422: Invalid user_id format (non-positive integer)
            - 401: Authentication required or invalid token
            - 403: Insufficient permissions to access user data
            - 500: Database connection error or internal server error
            
    Example:
        >>> response = await get_user(user_id=12345, include_profile=True)
        >>> print(response.name)
        "John Doe"
    """
    try:
        # Input validation beyond Pydantic
        if user_id <= 0:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={
                    "error": "INVALID_USER_ID",
                    "message": "User ID must be a positive integer",
                    "details": {"provided_user_id": user_id}
                }
            )
        
        # Business logic with proper error handling
        user = await user_service.get_user_by_id(user_id, include_profile=include_profile)
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "error": "USER_NOT_FOUND",
                    "message": f"User with ID {user_id} was not found",
                    "details": {"user_id": user_id}
                }
            )
        
        # Authorization check
        if not await user_service.can_access_user(current_user, user):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail={
                    "error": "ACCESS_DENIED",
                    "message": "Insufficient permissions to access this user's data",
                    "details": {"requested_user_id": user_id}
                }
            )
        
        return UserResponse(**user.dict())
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except DatabaseConnectionError as e:
        logger.error(f"Database connection failed for user {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "DATABASE_ERROR",
                "message": "Unable to connect to user database",
                "details": {"correlation_id": str(uuid.uuid4())}
            }
        )
    except Exception as e:
        logger.error(f"Unexpected error retrieving user {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "INTERNAL_ERROR",
                "message": "An unexpected error occurred while processing your request",
                "details": {"correlation_id": str(uuid.uuid4())}
            }
        )
```

### High-Quality Endpoint Sample
**Expected Confidence Score**: 0.80-0.90  
**Expected Rule Passes**: Good documentation, minor response model gaps

```python
# Golden Sample: FastAPI Endpoint - High Quality
@app.post("/users", response_model=UserResponse, status_code=201)
async def create_user(user_data: UserCreate) -> UserResponse:
    """
    Create a new user account with the provided information.
    
    Args:
        user_data: User creation data including email, name, and password
        
    Returns:
        UserResponse: The created user's information
        
    Raises:
        HTTPException: 400 if email already exists, 422 for validation errors
    """
    try:
        existing_user = await user_service.get_user_by_email(user_data.email)
        if existing_user:
            raise HTTPException(
                status_code=400,
                detail="User with this email already exists"
            )
        
        user = await user_service.create_user(user_data)
        return UserResponse(**user.dict())
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating user: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
```

### Moderate Quality Sample
**Expected Confidence Score**: 0.60-0.75  
**Expected Rule Passes**: Basic functionality, limited documentation

```python
# Golden Sample: FastAPI Endpoint - Moderate Quality
@app.get("/users/{user_id}")
async def get_user(user_id: int):
    """Get user by ID."""
    user = await user_service.get_user(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user
```

---

## COBOL Golden Samples

### Perfect Copybook Sample
**Expected Confidence Score**: 0.95-1.0  
**Expected Rule Passes**: All documentation, structure, and validation rules

```cobol
      *****************************************************************
      * COPYBOOK: CUSTOMER-RECORD                                    *
      * PURPOSE:  Customer master record layout for customer         *
      *           management system with comprehensive field          *
      *           definitions and business validation rules           *
      *                                                               *
      * AUTHOR:   Platform Team                                       *
      * DATE:     2023-01-15                                          *
      * VERSION:  3.2.1                                               *
      *                                                               *
      * DESCRIPTION:                                                  *
      *   This copybook defines the customer master record structure *
      *   used across all customer-related programs. Contains both   *
      *   demographic data and account relationship information.      *
      *                                                               *
      * BUSINESS RULES:                                               *
      *   - Customer ID must be 8-digit numeric                      *
      *   - SSN validation follows IRS Publication 15                *
      *   - Account status must be A, I, or C                        *
      *   - Credit limit cannot exceed $999,999.99                   *
      *                                                               *
      * CHANGE HISTORY:                                               *
      *   3.2.1 - 2023-01-15 - Added credit score field             *
      *   3.2.0 - 2022-12-01 - Enhanced address validation          *
      *   3.1.0 - 2022-06-15 - Added account relationship fields    *
      *                                                               *
      * CROSS-REFERENCES:                                             *
      *   Programs: CUSTMNT, CUSTINQ, CUSTRPT                        *
      *   Tables:   CUSTOMER_MASTER, CUSTOMER_ACCOUNTS               *
      *   Forms:    CRM001, CRM002                                    *
      *****************************************************************
      
       01  CUSTOMER-RECORD.
      *    ============================================================
      *    CUSTOMER IDENTIFICATION SECTION
      *    ============================================================
           05  CUST-ID                PIC 9(8).
      *        Customer unique identifier - system generated
      *        Format: 8-digit sequential number starting at 10000001
      *        Cannot be zero or negative
      *        Must exist in CUSTOMER_MASTER table
      
           05  CUST-TYPE              PIC X(2).
      *        Customer type classification
      *        Valid values: 'IN' = Individual, 'CO' = Corporate
      *                      'NP' = Non-Profit, 'GO' = Government
      *        Used for pricing and compliance rules
      
           05  CUST-STATUS            PIC X(1).
      *        Current account status indicator
      *        Valid values: 'A' = Active, 'I' = Inactive, 'C' = Closed
      *        Default: 'A' for new customers
      *        Status changes must be logged to AUDIT_LOG
      
      *    ============================================================
      *    PERSONAL INFORMATION SECTION
      *    ============================================================
           05  CUSTOMER-NAME.
               10  FIRST-NAME         PIC X(20).
      *            Customer's legal first name
      *            Required for individual customers
      *            Must not contain numbers or special characters
      *            Used for statements and legal documents
      
               10  MIDDLE-INITIAL     PIC X(1).
      *            Middle initial (optional)
      *            Space if not provided
      
               10  LAST-NAME          PIC X(30).
      *            Customer's legal last name or company name
      *            Required for all customer types
      *            Used as primary sort key in reports
      
           05  SSN                    PIC 9(9).
      *        Social Security Number (individuals only)
      *        Format: 9-digit number without dashes
      *        Required for individual customers in US
      *        Must pass IRS validation algorithm
      *        Encrypted in database storage
      
           05  TAX-ID                 PIC X(12).
      *        Federal Tax ID for corporate customers
      *        Format: XX-XXXXXXX for EIN
      *        Required for corporate, non-profit, government
      *        Must be validated against IRS database
      
      *    ============================================================
      *    ADDRESS INFORMATION SECTION
      *    ============================================================
           05  CUSTOMER-ADDRESS.
               10  STREET-ADDRESS     PIC X(40).
      *            Primary street address line 1
      *            Required for all customers
      *            Used for statement mailing
      
               10  ADDRESS-LINE-2     PIC X(40).
      *            Optional address line 2 (apartment, suite, etc.)
      *            Space if not provided
      
               10  CITY               PIC X(25).
      *            City name
      *            Required for all customers
      *            Must be valid city for state/ZIP combination
      
               10  STATE-CODE         PIC X(2).
      *            US state code (2-character abbreviation)
      *            Must be valid USPS state code
      *            Used for tax calculations
      
               10  ZIP-CODE           PIC 9(5).
      *            5-digit ZIP code
      *            Required for US addresses
      *            Used for geographic analysis and routing
      
               10  ZIP-PLUS-4         PIC 9(4).
      *            ZIP+4 extension (optional)
      *            9999 if not provided
      *            Improves mail delivery accuracy
      
      *    ============================================================
      *    FINANCIAL INFORMATION SECTION
      *    ============================================================
           05  FINANCIAL-DATA.
               10  CREDIT-LIMIT       PIC 9(7)V99 COMP-3.
      *            Customer's approved credit limit
      *            Format: 7 digits with 2 decimal places
      *            Maximum value: $999,999.99
      *            Must be approved by credit department
      *            Zero for cash-only customers
      
               10  CURRENT-BALANCE    PIC S9(7)V99 COMP-3.
      *            Current account balance
      *            Format: Signed 7 digits with 2 decimals
      *            Negative indicates customer credit
      *            Updated real-time by transaction processing
      
               10  CREDIT-SCORE       PIC 9(3).
      *            FICO credit score (300-850)
      *            Updated monthly from credit bureau
      *            999 if score unavailable
      *            Used for credit limit adjustments
      
               10  PAYMENT-TERMS      PIC 9(2).
      *            Standard payment terms in days
      *            Common values: 00=COD, 15=Net15, 30=Net30
      *            Must match approved terms in PAYMENT_TERMS table
      
      *    ============================================================
      *    AUDIT AND CONTROL SECTION
      *    ============================================================
           05  AUDIT-FIELDS.
               10  CREATED-DATE       PIC 9(8).
      *            Date record was created (YYYYMMDD)
      *            Set automatically by system
      *            Cannot be modified after creation
      
               10  CREATED-BY         PIC X(8).
      *            User ID who created the record
      *            Must exist in USER_MASTER table
      *            Used for audit trail purposes
      
               10  LAST-UPDATED       PIC 9(8).
      *            Date of last update (YYYYMMDD)
      *            Updated automatically on any change
      *            Used for data freshness validation
      
               10  UPDATED-BY         PIC X(8).
      *            User ID who last updated the record
      *            Must exist in USER_MASTER table
      *            Required for change tracking
      
               10  RECORD-VERSION     PIC 9(5).
      *            Optimistic locking version number
      *            Incremented on each update
      *            Prevents concurrent update conflicts
      
      *    ============================================================
      *    RELATIONSHIP SECTION
      *    ============================================================
           05  RELATIONSHIP-DATA.
               10  PRIMARY-ACCOUNT    PIC 9(10).
      *            Primary account number for this customer
      *            Links to ACCOUNT_MASTER table
      *            Zero if no accounts exist yet
      
               10  ACCOUNT-COUNT      PIC 9(3).
      *            Total number of active accounts
      *            Updated by account maintenance programs
      *            Used for relationship analysis
      
               10  BRANCH-CODE        PIC X(4).
      *            Home branch office code
      *            Must exist in BRANCH_MASTER table
      *            Used for relationship assignment
      
               10  OFFICER-ID         PIC X(6).
      *            Assigned relationship officer
      *            Must exist in OFFICER_MASTER table
      *            Required for commercial customers
      
      *    ============================================================
      *    RESERVED FIELDS FOR FUTURE EXPANSION
      *    ============================================================
           05  FILLER                 PIC X(50).
      *        Reserved for future enhancements
      *        Must be spaces when not in use
      *        Total record length: 400 bytes
```

### High-Quality Data Division Sample
**Expected Confidence Score**: 0.80-0.90

```cobol
      *****************************************************************
      * DATA DIVISION - WORKING STORAGE SECTION                      *
      * PURPOSE: Working variables for customer inquiry program       *
      *****************************************************************
       WORKING-STORAGE SECTION.
       
       01  WS-PROGRAM-DATA.
           05  WS-PROGRAM-NAME        PIC X(8) VALUE 'CUSTINQ'.
           05  WS-VERSION             PIC X(5) VALUE '2.1.0'.
           05  WS-RETURN-CODE         PIC 9(2) VALUE ZERO.
      *        Program return code: 00=Success, 04=Warning, 08=Error
       
       01  WS-CUSTOMER-WORK-AREA.
           05  WS-CUSTOMER-FOUND      PIC X(1) VALUE 'N'.
      *        Flag indicating if customer was found: Y/N
           05  WS-RECORD-COUNT        PIC 9(5) VALUE ZERO.
      *        Number of records processed
```

---

## IRS Layout Golden Samples

### Perfect Field Layout Sample
**Expected Confidence Score**: 0.95-1.0

```
*******************************************************************************
* IRS INDIVIDUAL MASTER FILE (IMF) RECORD LAYOUT                             *
* PUBLICATION: IRS Publication 1220 (Rev. 01-2023)                           *
* PURPOSE: Federal tax return processing and taxpayer account management      *
* SECURITY: Confidential - PII/Tax Information - PROTECT UNDER IRC 6103      *
*******************************************************************************

RECORD TYPE: Individual Master File (IMF) - Primary Taxpayer Record
RECORD LENGTH: 550 positions (fixed-length ASCII)
RECORD FORMAT: Primary Key = SSN + Tax Year + Sequence Number

FIELD DEFINITIONS:
================================================================================

POSITION 001-009: Social Security Number (SSN)
    FIELD NAME: Primary_SSN
    FORMAT: 9(9) - Numeric, no dashes or spaces
    DESCRIPTION: Primary taxpayer's Social Security Number
    VALIDATION: Must pass SSN validation algorithm per IRS Publication 15
    REQUIRED: Yes - Cannot be zeros or all nines
    EXAMPLE: 123456789
    BUSINESS RULE: Must be unique within tax year for primary filers
    AUDIT TRAIL: Changes logged to SSN_CHANGE_LOG table
    COMPLIANCE: Protected under IRC Section 6103 - Disclosure restrictions apply

POSITION 010-013: Tax Year
    FIELD NAME: Tax_Year
    FORMAT: 9(4) - Four-digit year
    DESCRIPTION: Tax year for which the return was filed
    VALIDATION: Must be within valid processing range (current year - 7 to current year + 1)
    REQUIRED: Yes
    EXAMPLE: 2023
    BUSINESS RULE: Cannot be future dated beyond next tax year
    DEFAULT VALUE: Current processing year

POSITION 014-016: Sequence Number
    FIELD NAME: Record_Sequence
    FORMAT: 9(3) - Sequential counter
    DESCRIPTION: Sequence number for multiple records per SSN/Year combination
    VALIDATION: Must be 001-999, unique within SSN/Year
    REQUIRED: Yes
    DEFAULT VALUE: 001 for initial filing
    BUSINESS RULE: Incremented for amended returns and adjustments

POSITION 017-046: Taxpayer Last Name
    FIELD NAME: Primary_Last_Name
    FORMAT: X(30) - Alphanumeric with restrictions
    DESCRIPTION: Primary taxpayer's legal last name as reported on return
    VALIDATION: No leading/trailing spaces, valid characters only (A-Z, apostrophe, hyphen, space)
    REQUIRED: Yes
    EXAMPLE: SMITH-JOHNSON
    BUSINESS RULE: Must match Social Security Administration records for e-filing
    COMPLIANCE: Name control matching required for return processing

POSITION 047-066: Taxpayer First Name
    FIELD NAME: Primary_First_Name
    FORMAT: X(20) - Alphanumeric with restrictions
    DESCRIPTION: Primary taxpayer's legal first name as reported on return
    VALIDATION: No leading/trailing spaces, valid characters only
    REQUIRED: Yes
    EXAMPLE: JOHN
    BUSINESS RULE: Must be consistent with SSA records for automated processing

POSITION 067-067: Middle Initial
    FIELD NAME: Primary_Middle_Initial
    FORMAT: X(1) - Single character
    DESCRIPTION: Primary taxpayer's middle initial
    VALIDATION: Must be A-Z or space
    REQUIRED: No
    DEFAULT VALUE: Space if not provided
    EXAMPLE: A

POSITION 068-097: Spouse Last Name
    FIELD NAME: Spouse_Last_Name
    FORMAT: X(30) - Alphanumeric with restrictions
    DESCRIPTION: Spouse's legal last name for joint returns
    VALIDATION: Same as primary taxpayer validation rules
    REQUIRED: Only for filing status 2 (Married Filing Jointly)
    BUSINESS RULE: Must be blank for single filers
    COMPLIANCE: Subject to same IRC 6103 protections

POSITION 098-106: Spouse SSN
    FIELD NAME: Spouse_SSN
    FORMAT: 9(9) - Numeric
    DESCRIPTION: Spouse's Social Security Number for joint returns
    VALIDATION: Must pass SSN validation, cannot duplicate primary SSN
    REQUIRED: Only for filing status 2 (Married Filing Jointly)
    EXAMPLE: 987654321
    BUSINESS RULE: Must be valid and active SSN per SSA verification
    AUDIT TRAIL: Changes logged with justification codes

POSITION 107-109: Filing Status Code
    FIELD NAME: Filing_Status
    FORMAT: 9(3) - Numeric code
    DESCRIPTION: Tax return filing status classification
    VALIDATION: Must be valid IRS filing status code
    REQUIRED: Yes
    VALID VALUES: 
        001 = Single
        002 = Married Filing Jointly  
        003 = Married Filing Separately
        004 = Head of Household
        005 = Qualifying Widow(er)
    EXAMPLE: 002
    BUSINESS RULE: Must be consistent with taxpayer and spouse information

POSITION 110-124: Adjusted Gross Income (AGI)
    FIELD NAME: Adjusted_Gross_Income
    FORMAT: 9(13)V99 - Signed numeric with 2 decimal places
    DESCRIPTION: Taxpayer's Adjusted Gross Income from Form 1040 Line 11
    VALIDATION: Must be within reasonable range (-99,999,999,999.99 to 99,999,999,999.99)
    REQUIRED: Yes
    EXAMPLE: 0000007500000 (represents $75,000.00)
    BUSINESS RULE: Used for automated processing and verification
    COMPLIANCE: Critical field for tax liability calculations

POSITION 125-139: Total Tax Liability
    FIELD NAME: Total_Tax_Liability  
    FORMAT: 9(13)V99 - Signed numeric with 2 decimal places
    DESCRIPTION: Total tax liability from Form 1040 Line 24
    VALIDATION: Cannot be negative, must be consistent with AGI and deductions
    REQUIRED: Yes
    EXAMPLE: 0000001125000 (represents $11,250.00)
    BUSINESS RULE: Validated against tax tables and calculation rules

POSITION 140-154: Federal Tax Withheld
    FIELD NAME: Federal_Tax_Withheld
    FORMAT: 9(13)V99 - Numeric with 2 decimal places
    DESCRIPTION: Total federal income tax withheld from all sources
    VALIDATION: Cannot be negative, must not exceed reasonable percentage of income
    REQUIRED: No, but expected if wage income reported
    EXAMPLE: 0000001000000 (represents $10,000.00)
    BUSINESS RULE: Validated against W-2 and 1099 information documents

POSITION 155-169: Refund Amount
    FIELD NAME: Refund_Amount
    FORMAT: 9(13)V99 - Numeric with 2 decimal places
    DESCRIPTION: Refund amount to be issued to taxpayer
    VALIDATION: Cannot be negative, must equal overpayment calculation
    REQUIRED: Only if refund due
    EXAMPLE: 0000000275000 (represents $2,750.00)
    BUSINESS RULE: Cannot exceed total payments minus tax liability

POSITION 170-184: Amount Owed
    FIELD NAME: Amount_Owed
    FORMAT: 9(13)V99 - Numeric with 2 decimal places  
    DESCRIPTION: Additional tax owed by taxpayer
    VALIDATION: Cannot be negative, must equal underpayment calculation
    REQUIRED: Only if additional tax due
    EXAMPLE: 0000000050000 (represents $500.00)
    BUSINESS RULE: Subject to interest and penalty calculations

POSITION 185-192: Return Due Date
    FIELD NAME: Return_Due_Date
    FORMAT: 9(8) - YYYYMMDD format
    DESCRIPTION: Legal due date for tax return (typically April 15)
    VALIDATION: Must be valid date, typically April 15 of following year
    REQUIRED: Yes
    EXAMPLE: 20240415
    BUSINESS RULE: Used for penalty and interest calculations

POSITION 193-200: Return Received Date
    FIELD NAME: Return_Received_Date
    FORMAT: 9(8) - YYYYMMDD format
    DESCRIPTION: Date return was received by IRS
    VALIDATION: Must be valid date, cannot be before tax year or future dated
    REQUIRED: Yes - Set by system upon receipt
    EXAMPLE: 20240315
    BUSINESS RULE: Used to determine timely filing for penalty calculations

POSITION 201-550: [Additional fields continue with same detailed format...]

*******************************************************************************
* VALIDATION RULES AND BUSINESS LOGIC                                        *
*******************************************************************************

CROSS-FIELD VALIDATIONS:
- If Filing_Status = 002 (MFJ), then Spouse_SSN and Spouse_Last_Name required
- If Refund_Amount > 0, then Amount_Owed must be zero
- If Amount_Owed > 0, then Refund_Amount must be zero
- Total_Tax_Liability must be consistent with AGI and filing status

DATA QUALITY RULES:
- No field may contain all zeros unless explicitly allowed
- Alphanumeric fields must not contain control characters
- Numeric fields must be right-justified with leading zeros
- Date fields must represent valid calendar dates

SECURITY AND COMPLIANCE:
- All fields containing PII subject to IRC Section 6103 disclosure restrictions
- Access logged and monitored per IRS security protocols
- Data retention per IRS Records Disposition Schedule
- Encryption required for data transmission and storage

AUDIT REQUIREMENTS:
- All changes must include user ID, timestamp, and reason code
- Original values preserved in audit trail tables
- Management approval required for sensitive field changes
- Quarterly data integrity validation reports required
```

---

## MUMPS/FileMan Golden Samples

### Perfect FileMan Data Dictionary Sample
**Expected Confidence Score**: 0.95-1.0

```
================================================================================
FILEMAN DATA DICTIONARY ENTRY - PATIENT FILE (#2)
================================================================================
VistA FileMan Version 22.2 - VA Medical Center Database
Last Updated: JAN 15, 2023@14:30:25
Security Classification: CONFIDENTIAL - Contains Protected Health Information (PHI)

FILE INFORMATION:
    File Number: 2
    File Name: PATIENT  
    Global Reference: ^DPT(
    Description: Master patient demographics and identification file containing
                 comprehensive patient information for all registered patients
                 in the VistA medical system.
    
    Purpose: Central repository for patient demographic data, insurance information,
             emergency contacts, and clinical flags. Used by all VistA applications
             requiring patient identification and demographic information.
    
    Security Requirements: 
        - HIPAA compliant access controls
        - Audit trail for all PHI access
        - Role-based permissions required
        - VPN access only for remote users

================================================================================
FIELD DEFINITIONS:
================================================================================

FIELD #: .01
FIELD NAME: NAME
FIELD TYPE: FREE TEXT (Required)
DESCRIPTION: Patient's full legal name as recorded in official documentation
INPUT TRANSFORM: K:$L(X)>30!($L(X)<3)!'(X'?1P.E) X
HELP PROMPT: Enter the patient's full legal name (Last,First Middle)
             Name must be 3-30 characters and cannot start with punctuation
AUDIT: YES - All changes logged with user ID and timestamp
CROSS-REFERENCE: 
    B: Regular cross-reference for lookup by name
    C: Soundex cross-reference for phonetic matching
    APCA: Patient Care Activity cross-reference
BUSINESS RULES:
    - Must be in LAST,FIRST MIDDLE format
    - Cannot contain numbers or special characters except comma, apostrophe, hyphen
    - Must be unique when combined with DOB and SSN
    - Changes require supervisor approval for established patients
EXAMPLES: 
    SMITH,JOHN WILLIAM
    O'CONNOR,MARY ELIZABETH
    GARCIA-LOPEZ,CARLOS

FIELD #: .02  
FIELD NAME: SEX
FIELD TYPE: SET OF CODES (Required)
DESCRIPTION: Patient's biological sex as recorded on birth certificate
SET OF CODES: 
    M:MALE
    F:FEMALE
    U:UNKNOWN
INPUT TRANSFORM: S X=$$UP^XLFSTR(X) K:'(X?1(1"M",1"F",1"U")) X
HELP PROMPT: Enter M for Male, F for Female, or U if Unknown
AUDIT: YES - Changes logged and require clinical justification
BUSINESS RULES:
    - Cannot be changed without documentation of clinical necessity
    - Used for dosing calculations and clinical decision support
    - Required for all patient registrations
CLINICAL IMPACT: Used by CPRS for medication dosing and clinical alerts

FIELD #: .03
FIELD NAME: DATE OF BIRTH  
FIELD TYPE: DATE (Required)
DESCRIPTION: Patient's date of birth as documented on birth certificate or official ID
INPUT TRANSFORM: S %DT="EX" D ^%DT S X=Y K:Y<1 X
HELP PROMPT: Enter date of birth (MM/DD/YYYY or MM/DD/YY)
             Must be a valid past date
AUDIT: YES - Critical field requiring supervisor approval for changes
CROSS-REFERENCE:
    BS: Cross-reference for age-based reports
    ADOBUCI: DoB cross-reference for duplicate checking
BUSINESS RULES:
    - Must be in the past (cannot be future dated)
    - Cannot be more than 150 years ago
    - Used in duplicate patient checking algorithm
    - Required for Medicare and insurance billing
    - Changes require supporting documentation
CLINICAL IMPACT: 
    - Critical for medication dosing calculations
    - Used for age-appropriate clinical decision support
    - Required for pediatric and geriatric protocols

FIELD #: .09
FIELD NAME: SOCIAL SECURITY NUMBER
FIELD TYPE: FREE TEXT
DESCRIPTION: Patient's 9-digit Social Security Number without dashes
INPUT TRANSFORM: K:X'?9N!(X?9"0")!($E(X,1,3)="000")!($E(X,4,5)="00") X
HELP PROMPT: Enter 9-digit Social Security Number (no dashes)
             Format: 123456789
AUDIT: YES - All access and changes logged per HIPAA requirements
SECURITY: 
    - Encrypted storage required
    - Access restricted to authorized personnel only
    - Display masked except last 4 digits for most users
BUSINESS RULES:
    - Must be 9 numeric digits
    - Cannot be all zeros or invalid SSN patterns
    - Used for duplicate patient detection
    - Required for Medicare and most insurance claims
    - Must be verified against official documentation
COMPLIANCE:
    - HIPAA: Requires minimum necessary standard
    - Privacy Act: Subject to disclosure restrictions
    - IRC 6103: Tax information protection applies

FIELD #: .111
FIELD NAME: STREET ADDRESS [LINE 1]
FIELD TYPE: FREE TEXT
DESCRIPTION: Primary street address for patient correspondence and emergency contact
INPUT TRANSFORM: K:$L(X)>35!($L(X)<5) X
HELP PROMPT: Enter street address (house number and street name)
             Must be 5-35 characters
AUDIT: NO - Non-clinical administrative data
BUSINESS RULES:
    - Required for outpatient scheduling and billing
    - Used for geographic analysis and catchment area reporting
    - Validated against USPS address database when possible
    - Updated during patient registration process

FIELD #: .112
FIELD NAME: STREET ADDRESS [LINE 2]  
FIELD TYPE: FREE TEXT
DESCRIPTION: Additional address information (apartment, suite, unit number)
INPUT TRANSFORM: K:$L(X)>35 X
HELP PROMPT: Enter apartment, suite, or unit number (optional)
AUDIT: NO
BUSINESS RULES:
    - Optional field for additional address details
    - Combined with Line 1 for complete mailing address
    - Used for appointment reminder mailings

FIELD #: .115
FIELD NAME: CITY
FIELD TYPE: FREE TEXT (Required for billing)
DESCRIPTION: City name for patient's primary address
INPUT TRANSFORM: K:$L(X)>20!($L(X)<2) X
HELP PROMPT: Enter city name (2-20 characters)
AUDIT: NO
BUSINESS RULES:
    - Required for billing and correspondence
    - Must be valid city name for state
    - Used in geographic reporting and analysis

FIELD #: .117
FIELD NAME: STATE
FIELD TYPE: POINTER TO STATE FILE (#5)
DESCRIPTION: US state or territory for patient's primary address
INPUT TRANSFORM: State must exist in State File and be currently active
HELP PROMPT: Enter state name or abbreviation
AUDIT: NO
POINTER REFERENCE: Points to STATE file (#5)
BUSINESS RULES:
    - Must be valid US state, territory, or military designation
    - Required for billing and regulatory reporting
    - Used for interstate care coordination
    - Validates against official USPS state codes

FIELD #: .134
FIELD NAME: PHONE NUMBER [RESIDENCE]
FIELD TYPE: FREE TEXT
DESCRIPTION: Patient's primary residence telephone number
INPUT TRANSFORM: K:'(X?10N) X
HELP PROMPT: Enter 10-digit phone number (area code + number)
             Format: 1234567890
AUDIT: NO
BUSINESS RULES:
    - Must be 10 digits (area code required)
    - Used for appointment reminders and emergency contact
    - Validated for US/Canada phone number format
    - Multiple phone numbers can be stored in subfields

FIELD #: .361
FIELD NAME: EMERGENCY CONTACT
FIELD TYPE: MULTIPLE SUBFILE (#2.361)
DESCRIPTION: Emergency contact information including name, relationship, and phone
SUBFILE STRUCTURE:
    Field .01: CONTACT NAME (Free Text, Required)
    Field .02: RELATIONSHIP (Pointer to RELATIONSHIP file)
    Field .03: PHONE NUMBER (Free Text, 10 digits)
    Field .04: ALTERNATE PHONE (Free Text, optional)
INPUT TRANSFORM: Contact name required, phone must be 10 digits
HELP PROMPT: Enter emergency contact information
AUDIT: YES - Critical for patient safety
BUSINESS RULES:
    - At least one emergency contact required for inpatients
    - Multiple contacts allowed, prioritized by entry order
    - Used by clinical staff for family notification
    - Required for surgical procedures and high-risk treatments
CLINICAL IMPACT: Critical for emergency notifications and family contact

================================================================================
CLINICAL DECISION SUPPORT INTEGRATION:
================================================================================

ALLERGY CHECKING:
    - Patient allergies cross-referenced with medication orders
    - Automated alerts generated for potential adverse reactions
    - Integration with CPRS order checking

DUPLICATE DETECTION:
    - Multi-field algorithm using name, DOB, SSN, and address
    - Soundex matching for similar names
    - Manual review process for potential duplicates

AGE-BASED PROTOCOLS:
    - Pediatric dosing calculations based on DOB
    - Geriatric medication interaction checking
    - Age-appropriate clinical guidelines activation

================================================================================
HIPAA COMPLIANCE AND AUDIT REQUIREMENTS:
================================================================================

ACCESS CONTROLS:
    - Role-based access with minimum necessary principle
    - User authentication and authorization required
    - Access logging for all PHI fields

AUDIT TRAIL:
    - All changes to critical fields logged with timestamp and user
    - Quarterly audit reports for data integrity
    - Annual HIPAA compliance review

PATIENT RIGHTS:
    - Patients can request access to their demographic information
    - Amendment process for incorrect information
    - Accounting of disclosures maintained

BREACH NOTIFICATION:
    - Unauthorized access triggers automatic incident reporting
    - Risk assessment and notification procedures per HIPAA
    - Integration with VA privacy office protocols

================================================================================
TECHNICAL SPECIFICATIONS:
================================================================================

STORAGE: Global array structure ^DPT(IEN,field)
INDEXING: Multiple cross-references for efficient lookup
BACKUP: Daily incremental, weekly full backup
PERFORMANCE: Sub-second response for single patient lookup
INTEGRATION: Real-time HL7 ADT messaging to clinical systems
DISASTER RECOVERY: 24-hour recovery time objective
```

---

## Quality Score Correlations

### Confidence Score Ranges by Quality Level

#### Perfect Quality (0.95-1.0)
- **All required fields present and complete**
- **Comprehensive documentation with business context**
- **Proper formatting and consistent style**
- **Domain-specific vocabulary usage**
- **Security and compliance considerations addressed**
- **Examples and validation rules provided**
- **Cross-references and relationships documented**

#### High Quality (0.80-0.95)
- **Most required fields present**
- **Good documentation with clear descriptions**
- **Proper formatting**
- **Some domain vocabulary**
- **Basic security considerations**
- **Limited examples provided**

#### Moderate Quality (0.60-0.80)
- **Basic required fields present**
- **Minimal documentation**
- **Acceptable formatting**
- **Limited domain vocabulary**
- **Security not addressed**
- **No examples**

#### Low Quality (0.40-0.60)
- **Some required fields missing**
- **Poor or missing documentation**
- **Inconsistent formatting**
- **No domain vocabulary**
- **Major gaps in completeness**

#### Very Low Quality (0.0-0.40)
- **Many required fields missing**
- **No meaningful documentation**
- **Poor structure**
- **Placeholder text present**
- **Critical information missing**

---

## Usage Guidelines

### For Rule Development
1. **Test against golden samples** to verify rule logic
2. **Use confidence ranges** to calibrate scoring algorithms  
3. **Validate template output** against expected suggestions
4. **Check severity escalation** for security/compliance content

### For System Validation
1. **Benchmark performance** against known-good data
2. **Verify consistency** across artifact types
3. **Test edge cases** with modified golden samples
4. **Validate output format** meets requirements

### For Team Training
1. **Share samples** as documentation standards
2. **Use in code reviews** as quality checkpoints
3. **Reference in guidelines** for new developers
4. **Update samples** as standards evolve

These golden samples provide concrete examples of documentation quality standards and serve as reliable test fixtures for the gap analysis system across all supported artifact types.