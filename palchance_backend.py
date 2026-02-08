"""
PalChance Backend - Fully Compatible with React Frontend
FastAPI + AI Analysis + Smart Matching
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Query, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl, Field, EmailStr
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime, Text, JSON, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
import uvicorn
import json
import os
from collections import Counter
import re

# ============= Database Setup =============

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./palchance.db")

engine = create_engine(
    DATABASE_URL, 
    echo=False,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# ============= Database Models =============

class User(Base):
    """Users table"""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True)
    full_name = Column(String(255))
    phone = Column(String(50), nullable=True)
    location = Column(String(100), nullable=True)
    skills = Column(JSON, default=list)
    bio = Column(Text, nullable=True)
    experience_years = Column(Float, default=0.0)
    education = Column(JSON, default=list)
    preferred_job_type = Column(String(50), nullable=True)
    preferred_work_mode = Column(String(50), nullable=True)
    is_verified = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.now)
    
    # Relationships
    applications = relationship("Application", back_populates="user")
    saved_jobs = relationship("SavedJob", back_populates="user")


class Company(Base):
    """Companies table"""
    __tablename__ = "companies"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), index=True)
    description = Column(Text, nullable=True)
    industry = Column(String(100), nullable=True)
    size = Column(String(50), nullable=True)
    location = Column(String(100), nullable=True)
    website = Column(String(500), nullable=True)
    logo_url = Column(String(500), nullable=True)
    is_verified = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.now)
    
    # Relationships
    opportunities = relationship("Opportunity", back_populates="company")


class Opportunity(Base):
    """Job opportunities table"""
    __tablename__ = "opportunities"
    
    id = Column(Integer, primary_key=True, index=True)
    company_id = Column(Integer, ForeignKey('companies.id'), nullable=True)
    title = Column(String(255), index=True)
    description = Column(Text)
    job_type = Column(String(50), index=True)  # internship, job, training
    work_mode = Column(String(50), nullable=True)  # remote, on-site, hybrid
    location = Column(String(100), nullable=True, index=True)
    salary_min = Column(Float, nullable=True)
    salary_max = Column(Float, nullable=True)
    currency = Column(String(10), default="ILS")
    experience_level = Column(String(50), nullable=True)  # entry, mid, senior
    skills_required = Column(JSON, default=list)
    requirements = Column(JSON, default=list)
    deadline = Column(DateTime, nullable=True)
    contact_email = Column(String(255), nullable=True)
    contact_phone = Column(String(50), nullable=True)
    source_url = Column(String(500), nullable=True)
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)
    views_count = Column(Integer, default=0)
    applications_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.now, index=True)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    
    # Relationships
    company = relationship("Company", back_populates="opportunities")
    applications = relationship("Application", back_populates="opportunity")
    saved_by = relationship("SavedJob", back_populates="opportunity")


class Application(Base):
    """Job applications table"""
    __tablename__ = "applications"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey('users.id'), index=True)
    opportunity_id = Column(Integer, ForeignKey('opportunities.id'), index=True)
    status = Column(String(50), default="pending", index=True)
    cover_letter = Column(Text, nullable=True)
    resume_url = Column(String(500), nullable=True)
    match_score = Column(Float, nullable=True)
    applied_at = Column(DateTime, default=datetime.now)
    updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
    
    # Relationships
    user = relationship("User", back_populates="applications")
    opportunity = relationship("Opportunity", back_populates="applications")


class SavedJob(Base):
    """Saved jobs table"""
    __tablename__ = "saved_jobs"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey('users.id'), index=True)
    opportunity_id = Column(Integer, ForeignKey('opportunities.id'), index=True)
    saved_at = Column(DateTime, default=datetime.now)
    
    # Relationships
    user = relationship("User", back_populates="saved_jobs")
    opportunity = relationship("Opportunity", back_populates="saved_by")


# Create tables
Base.metadata.create_all(bind=engine)


# ============= Pydantic Models (API Schemas) =============

class UserCreate(BaseModel):
    email: EmailStr
    full_name: str
    phone: Optional[str] = None
    location: Optional[str] = None
    skills: List[str] = []
    bio: Optional[str] = None


class UserUpdate(BaseModel):
    full_name: Optional[str] = None
    phone: Optional[str] = None
    location: Optional[str] = None
    skills: Optional[List[str]] = None
    bio: Optional[str] = None
    experience_years: Optional[float] = None
    education: Optional[List[Dict]] = None
    preferred_job_type: Optional[str] = None
    preferred_work_mode: Optional[str] = None


class UserResponse(BaseModel):
    id: int
    email: str
    full_name: str
    phone: Optional[str]
    location: Optional[str]
    skills: List[str]
    bio: Optional[str]
    experience_years: float
    is_verified: bool
    created_at: datetime
    
    class Config:
        from_attributes = True


class OpportunityCreate(BaseModel):
    title: str
    description: str
    job_type: str
    work_mode: Optional[str] = None
    location: Optional[str] = None
    salary_min: Optional[float] = None
    salary_max: Optional[float] = None
    experience_level: Optional[str] = None
    skills_required: List[str] = []
    requirements: List[str] = []
    deadline: Optional[datetime] = None
    contact_email: Optional[str] = None
    company_name: Optional[str] = None


class OpportunityResponse(BaseModel):
    id: int
    title: str
    description: str
    job_type: str
    work_mode: Optional[str]
    location: Optional[str]
    salary_min: Optional[float]
    salary_max: Optional[float]
    currency: str
    experience_level: Optional[str]
    skills_required: List[str]
    requirements: List[str]
    deadline: Optional[datetime]
    contact_email: Optional[str]
    is_active: bool
    is_verified: bool
    views_count: int
    applications_count: int
    created_at: datetime
    company_name: Optional[str] = None
    match_score: Optional[float] = None
    
    class Config:
        from_attributes = True


class ApplicationCreate(BaseModel):
    opportunity_id: int
    cover_letter: Optional[str] = None


class ApplicationResponse(BaseModel):
    id: int
    user_id: int
    opportunity_id: int
    status: str
    match_score: Optional[float]
    applied_at: datetime
    opportunity: Optional[OpportunityResponse] = None
    
    class Config:
        from_attributes = True


class OpportunityFilter(BaseModel):
    job_type: Optional[str] = None
    location: Optional[str] = None
    work_mode: Optional[str] = None
    experience_level: Optional[str] = None
    skills: Optional[List[str]] = None
    min_salary: Optional[float] = None
    max_salary: Optional[float] = None


class MatchResult(BaseModel):
    total_score: float
    skills_match: float
    location_match: float
    experience_match: float
    recommendations: List[str]


# ============= AI Matching Engine =============

class SimpleMatchingEngine:
    """Simple matching algorithm for job-candidate matching"""
    
    def calculate_match(
        self,
        user_skills: List[str],
        user_location: Optional[str],
        user_experience: float,
        opportunity: Opportunity
    ) -> MatchResult:
        """Calculate match score between user and opportunity"""
        
        total_score = 0.0
        recommendations = []
        
        # Skills matching (60% weight)
        skills_score = self._match_skills(user_skills, opportunity.skills_required)
        total_score += skills_score * 0.6
        
        if skills_score < 70:
            missing = set(opportunity.skills_required) - set(user_skills)
            if missing:
                recommendations.append(f"Consider learning: {', '.join(list(missing)[:3])}")
        
        # Location matching (20% weight)
        location_score = self._match_location(user_location, opportunity.location)
        total_score += location_score * 0.2
        
        # Experience matching (20% weight)
        experience_score = self._match_experience(user_experience, opportunity.experience_level)
        total_score += experience_score * 0.2
        
        if total_score >= 80:
            recommendations.append("Excellent match! Apply now!")
        elif total_score >= 60:
            recommendations.append("Good match. Review requirements carefully.")
        else:
            recommendations.append("Consider gaining more relevant experience.")
        
        return MatchResult(
            total_score=round(total_score, 1),
            skills_match=round(skills_score, 1),
            location_match=round(location_score, 1),
            experience_match=round(experience_score, 1),
            recommendations=recommendations
        )
    
    def _match_skills(self, user_skills: List[str], required_skills: List[str]) -> float:
        """Match skills"""
        if not required_skills:
            return 70.0
        if not user_skills:
            return 0.0
        
        user_lower = [s.lower() for s in user_skills]
        required_lower = [s.lower() for s in required_skills]
        
        matches = sum(1 for req in required_lower if req in user_lower)
        return (matches / len(required_skills)) * 100
    
    def _match_location(self, user_location: Optional[str], job_location: Optional[str]) -> float:
        """Match location"""
        if not job_location:
            return 70.0
        if not user_location:
            return 50.0
        
        if user_location.lower() in job_location.lower() or job_location.lower() in user_location.lower():
            return 100.0
        return 30.0
    
    def _match_experience(self, user_years: float, required_level: Optional[str]) -> float:
        """Match experience"""
        if not required_level:
            return 70.0
        
        level_map = {
            'entry': (0, 2),
            'mid': (2, 5),
            'senior': (5, 100)
        }
        
        min_years, max_years = level_map.get(required_level, (0, 100))
        
        if min_years <= user_years <= max_years:
            return 100.0
        elif user_years < min_years:
            return max(0, 100 - ((min_years - user_years) * 20))
        else:
            return 90.0


# ============= Dependency =============

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ============= FastAPI App =============

app = FastAPI(
    title="PalChance API",
    description="Smart Job Platform for Palestine",
    version="1.0.0"
)

# CORS - Allow React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Vite default
        "http://localhost:3000",  # Alternative
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize matching engine
matching_engine = SimpleMatchingEngine()


# ============= API Endpoints =============

@app.get("/")
async def root():
    """API root"""
    return {
        "name": "PalChance API",
        "version": "1.0.0",
        "status": "active",
        "endpoints": {
            "users": "/api/users",
            "opportunities": "/api/opportunities",
            "applications": "/api/applications"
        }
    }


@app.get("/api/health")
async def health_check():
    """Health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }


# ============= User Endpoints =============

@app.post("/api/users", response_model=UserResponse, status_code=201)
async def create_user(user_data: UserCreate, db: Session = Depends(get_db)):
    """Create new user"""
    # Check if email exists
    existing = db.query(User).filter(User.email == user_data.email).first()
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    user = User(**user_data.dict())
    db.add(user)
    db.commit()
    db.refresh(user)
    
    return user


@app.get("/api/users/{user_id}", response_model=UserResponse)
async def get_user(user_id: int, db: Session = Depends(get_db)):
    """Get user by ID"""
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user


@app.put("/api/users/{user_id}", response_model=UserResponse)
async def update_user(
    user_id: int,
    user_data: UserUpdate,
    db: Session = Depends(get_db)
):
    """Update user"""
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    for key, value in user_data.dict(exclude_unset=True).items():
        setattr(user, key, value)
    
    db.commit()
    db.refresh(user)
    return user


# ============= Opportunity Endpoints =============

@app.post("/api/opportunities", response_model=OpportunityResponse, status_code=201)
async def create_opportunity(
    opp_data: OpportunityCreate,
    db: Session = Depends(get_db)
):
    """Create new opportunity"""
    
    # Handle company
    company = None
    if opp_data.company_name:
        company = db.query(Company).filter(Company.name == opp_data.company_name).first()
        if not company:
            company = Company(name=opp_data.company_name, is_verified=False)
            db.add(company)
            db.commit()
            db.refresh(company)
    
    # Create opportunity
    opp_dict = opp_data.dict()
    opp_dict.pop('company_name', None)
    
    opportunity = Opportunity(
        **opp_dict,
        company_id=company.id if company else None
    )
    
    db.add(opportunity)
    db.commit()
    db.refresh(opportunity)
    
    # Add company name to response
    result = OpportunityResponse.from_orm(opportunity)
    if company:
        result.company_name = company.name
    
    return result


@app.get("/api/opportunities", response_model=List[OpportunityResponse])
async def get_opportunities(
    user_id: Optional[int] = None,
    job_type: Optional[str] = None,
    location: Optional[str] = None,
    work_mode: Optional[str] = None,
    skip: int = 0,
    limit: int = 50,
    db: Session = Depends(get_db)
):
    """Get all opportunities with optional filters and matching"""
    
    query = db.query(Opportunity).filter(Opportunity.is_active == True)
    
    # Apply filters
    if job_type:
        query = query.filter(Opportunity.job_type == job_type)
    if location:
        query = query.filter(Opportunity.location.ilike(f"%{location}%"))
    if work_mode:
        query = query.filter(Opportunity.work_mode == work_mode)
    
    opportunities = query.order_by(Opportunity.created_at.desc()).offset(skip).limit(limit).all()
    
    # Calculate match scores if user_id provided
    results = []
    for opp in opportunities:
        opp_dict = OpportunityResponse.from_orm(opp).dict()
        
        # Add company name
        if opp.company:
            opp_dict['company_name'] = opp.company.name
        
        # Calculate match score
        if user_id:
            user = db.query(User).filter(User.id == user_id).first()
            if user:
                match = matching_engine.calculate_match(
                    user.skills or [],
                    user.location,
                    user.experience_years,
                    opp
                )
                opp_dict['match_score'] = match.total_score
        
        results.append(opp_dict)
    
    # Sort by match score if available
    if user_id:
        results.sort(key=lambda x: x.get('match_score', 0), reverse=True)
    
    return results


@app.get("/api/opportunities/{opp_id}", response_model=OpportunityResponse)
async def get_opportunity(
    opp_id: int,
    user_id: Optional[int] = None,
    db: Session = Depends(get_db)
):
    """Get single opportunity"""
    opp = db.query(Opportunity).filter(Opportunity.id == opp_id).first()
    if not opp:
        raise HTTPException(status_code=404, detail="Opportunity not found")
    
    # Increment views
    opp.views_count += 1
    db.commit()
    
    opp_dict = OpportunityResponse.from_orm(opp).dict()
    
    # Add company name
    if opp.company:
        opp_dict['company_name'] = opp.company.name
    
    # Calculate match score
    if user_id:
        user = db.query(User).filter(User.id == user_id).first()
        if user:
            match = matching_engine.calculate_match(
                user.skills or [],
                user.location,
                user.experience_years,
                opp
            )
            opp_dict['match_score'] = match.total_score
    
    return opp_dict


@app.post("/api/opportunities/search", response_model=List[OpportunityResponse])
async def search_opportunities(
    filters: OpportunityFilter,
    user_id: Optional[int] = None,
    skip: int = 0,
    limit: int = 50,
    db: Session = Depends(get_db)
):
    """Search opportunities with filters"""
    
    query = db.query(Opportunity).filter(Opportunity.is_active == True)
    
    if filters.job_type:
        query = query.filter(Opportunity.job_type == filters.job_type)
    if filters.location:
        query = query.filter(Opportunity.location.ilike(f"%{filters.location}%"))
    if filters.work_mode:
        query = query.filter(Opportunity.work_mode == filters.work_mode)
    if filters.experience_level:
        query = query.filter(Opportunity.experience_level == filters.experience_level)
    if filters.min_salary:
        query = query.filter(Opportunity.salary_min >= filters.min_salary)
    if filters.max_salary:
        query = query.filter(Opportunity.salary_max <= filters.max_salary)
    
    opportunities = query.order_by(Opportunity.created_at.desc()).offset(skip).limit(limit).all()
    
    results = []
    for opp in opportunities:
        opp_dict = OpportunityResponse.from_orm(opp).dict()
        if opp.company:
            opp_dict['company_name'] = opp.company.name
        
        if user_id:
            user = db.query(User).filter(User.id == user_id).first()
            if user:
                match = matching_engine.calculate_match(
                    user.skills or [],
                    user.location,
                    user.experience_years,
                    opp
                )
                opp_dict['match_score'] = match.total_score
        
        results.append(opp_dict)
    
    if user_id:
        results.sort(key=lambda x: x.get('match_score', 0), reverse=True)
    
    return results


# ============= Application Endpoints =============

@app.post("/api/applications", response_model=ApplicationResponse, status_code=201)
async def create_application(
    app_data: ApplicationCreate,
    user_id: int,
    db: Session = Depends(get_db)
):
    """Apply to opportunity"""
    
    # Check user exists
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Check opportunity exists
    opp = db.query(Opportunity).filter(Opportunity.id == app_data.opportunity_id).first()
    if not opp:
        raise HTTPException(status_code=404, detail="Opportunity not found")
    
    # Check not already applied
    existing = db.query(Application).filter(
        Application.user_id == user_id,
        Application.opportunity_id == app_data.opportunity_id
    ).first()
    
    if existing:
        raise HTTPException(status_code=400, detail="Already applied to this opportunity")
    
    # Calculate match score
    match = matching_engine.calculate_match(
        user.skills or [],
        user.location,
        user.experience_years,
        opp
    )
    
    # Create application
    application = Application(
        user_id=user_id,
        opportunity_id=app_data.opportunity_id,
        cover_letter=app_data.cover_letter,
        match_score=match.total_score
    )
    
    db.add(application)
    
    # Update count
    opp.applications_count += 1
    
    db.commit()
    db.refresh(application)
    
    return application


@app.get("/api/users/{user_id}/applications", response_model=List[ApplicationResponse])
async def get_user_applications(
    user_id: int,
    db: Session = Depends(get_db)
):
    """Get user's applications"""
    
    applications = db.query(Application).filter(
        Application.user_id == user_id
    ).order_by(Application.applied_at.desc()).all()
    
    results = []
    for app in applications:
        app_dict = ApplicationResponse.from_orm(app).dict()
        
        # Add opportunity details
        if app.opportunity:
            opp_dict = OpportunityResponse.from_orm(app.opportunity).dict()
            if app.opportunity.company:
                opp_dict['company_name'] = app.opportunity.company.name
            app_dict['opportunity'] = opp_dict
        
        results.append(app_dict)
    
    return results


@app.post("/api/users/{user_id}/saved-jobs/{opp_id}")
async def save_job(
    user_id: int,
    opp_id: int,
    db: Session = Depends(get_db)
):
    """Save a job"""
    
    # Check if already saved
    existing = db.query(SavedJob).filter(
        SavedJob.user_id == user_id,
        SavedJob.opportunity_id == opp_id
    ).first()
    
    if existing:
        return {"message": "Already saved"}
    
    saved = SavedJob(user_id=user_id, opportunity_id=opp_id)
    db.add(saved)
    db.commit()
    
    return {"message": "Job saved successfully"}


@app.delete("/api/users/{user_id}/saved-jobs/{opp_id}")
async def unsave_job(
    user_id: int,
    opp_id: int,
    db: Session = Depends(get_db)
):
    """Unsave a job"""
    
    saved = db.query(SavedJob).filter(
        SavedJob.user_id == user_id,
        SavedJob.opportunity_id == opp_id
    ).first()
    
    if saved:
        db.delete(saved)
        db.commit()
    
    return {"message": "Job removed from saved"}


@app.get("/api/users/{user_id}/saved-jobs", response_model=List[OpportunityResponse])
async def get_saved_jobs(
    user_id: int,
    db: Session = Depends(get_db)
):
    """Get user's saved jobs"""
    
    saved_jobs = db.query(SavedJob).filter(
        SavedJob.user_id == user_id
    ).all()
    
    results = []
    for saved in saved_jobs:
        if saved.opportunity and saved.opportunity.is_active:
            opp_dict = OpportunityResponse.from_orm(saved.opportunity).dict()
            if saved.opportunity.company:
                opp_dict['company_name'] = saved.opportunity.company.name
            results.append(opp_dict)
    
    return results


# ============= Statistics =============

@app.get("/api/stats")
async def get_statistics(db: Session = Depends(get_db)):
    """Get platform statistics"""
    
    total_opportunities = db.query(Opportunity).filter(Opportunity.is_active == True).count()
    total_users = db.query(User).count()
    total_applications = db.query(Application).count()
    total_companies = db.query(Company).filter(Company.is_verified == True).count()
    
    # By job type
    internships = db.query(Opportunity).filter(
        Opportunity.job_type == 'internship',
        Opportunity.is_active == True
    ).count()
    
    jobs = db.query(Opportunity).filter(
        Opportunity.job_type == 'job',
        Opportunity.is_active == True
    ).count()
    
    training = db.query(Opportunity).filter(
        Opportunity.job_type == 'training',
        Opportunity.is_active == True
    ).count()
    
    # Top locations
    locations = db.query(Opportunity.location).filter(
        Opportunity.location.isnot(None),
        Opportunity.is_active == True
    ).all()
    location_counts = Counter([loc[0] for loc in locations])
    top_locations = [
        {"location": loc, "count": count}
        for loc, count in location_counts.most_common(5)
    ]
    
    return {
        "overview": {
            "total_opportunities": total_opportunities,
            "total_users": total_users,
            "total_applications": total_applications,
            "verified_companies": total_companies
        },
        "by_type": {
            "internships": internships,
            "jobs": jobs,
            "training": training
        },
        "top_locations": top_locations
    }


# ============= Run Server =============

if __name__ == "__main__":
    print("=" * 50)
    print("PalChance Backend Server")
    print("=" * 50)
    print("Status: Starting...")
    print("URL: http://localhost:8000")
    print("Docs: http://localhost:8000/docs")
    print("=" * 50)
    
    uvicorn.run(
        "palchance_backend:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
