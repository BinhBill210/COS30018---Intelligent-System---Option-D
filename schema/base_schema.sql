-- Base Schema for Business Intelligence System
-- Extensible foundation that any tool can build upon

-- ========================================
-- BUSINESSES TABLE
-- ========================================
CREATE TABLE IF NOT EXISTS businesses (
    business_id VARCHAR PRIMARY KEY,
    name VARCHAR NOT NULL,
    address VARCHAR,
    city VARCHAR,
    state VARCHAR(2),
    postal_code VARCHAR(10),
    latitude DOUBLE,
    longitude DOUBLE,
    stars DOUBLE,
    review_count INTEGER DEFAULT 0,
    is_open BOOLEAN DEFAULT true,
    attributes TEXT,  -- JSON string for flexible attributes
    categories VARCHAR,  -- Comma-separated categories
    hours TEXT,  -- JSON string for hours
    
    -- Metadata for tracking
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ========================================
-- REVIEWS TABLE (Optional - for future tools)
-- ========================================
CREATE TABLE IF NOT EXISTS reviews (
    review_id VARCHAR PRIMARY KEY,
    user_id VARCHAR,
    business_id VARCHAR NOT NULL,
    stars DOUBLE NOT NULL,
    useful INTEGER DEFAULT 0,
    funny INTEGER DEFAULT 0,
    cool INTEGER DEFAULT 0,
    text TEXT,
    date TIMESTAMP,
    
    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Foreign key relationship
    FOREIGN KEY (business_id) REFERENCES businesses(business_id)
);

-- ========================================
-- BASIC INDEXES FOR PERFORMANCE
-- ========================================

-- Business indexes
CREATE INDEX IF NOT EXISTS idx_businesses_name ON businesses(name);
CREATE INDEX IF NOT EXISTS idx_businesses_city ON businesses(city);
CREATE INDEX IF NOT EXISTS idx_businesses_categories ON businesses(categories);
CREATE INDEX IF NOT EXISTS idx_businesses_stars ON businesses(stars);
CREATE INDEX IF NOT EXISTS idx_businesses_city_state ON businesses(city, state);

-- Review indexes (if reviews table is used)
CREATE INDEX IF NOT EXISTS idx_reviews_business_id ON reviews(business_id);
CREATE INDEX IF NOT EXISTS idx_reviews_stars ON reviews(stars);
CREATE INDEX IF NOT EXISTS idx_reviews_date ON reviews(date);

-- ========================================
-- UTILITY VIEWS (Optional)
-- ========================================

-- Business summary view (can be extended by future tools)
CREATE VIEW IF NOT EXISTS business_summary AS
SELECT 
    business_id,
    name,
    city,
    state,
    stars,
    review_count,
    categories,
    is_open
FROM businesses;

-- Sample queries that tools can build upon:
-- Fast business lookup: SELECT * FROM businesses WHERE LOWER(name) = LOWER(?);
-- City search: SELECT * FROM businesses WHERE LOWER(city) = LOWER(?) ORDER BY stars DESC;
-- Category search: SELECT * FROM businesses WHERE LOWER(categories) LIKE LOWER(?) ORDER BY stars DESC;
-- Geographic search: SELECT * FROM businesses WHERE latitude BETWEEN ? AND ? AND longitude BETWEEN ? AND ?;
