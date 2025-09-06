import pandas as pd
import re
from typing import Dict, List, Any, Optional
from collections import Counter


class ReviewResponseTool:
    """
    Tool for generating personalized responses to customer reviews.
    Follows the same modular pattern as other tools in the system.
    """

    def __init__(self, business_data_path: str = "data/processed/business_cleaned.parquet",
                 review_data_path: str = "data/processed/review_cleaned.parquet"):
        """Initialize with business and review data - consistent with other tools"""
        self.business_df = pd.read_parquet(business_data_path) if business_data_path.endswith(
            '.parquet') else pd.read_csv(business_data_path)
        self.review_df = pd.read_parquet(review_data_path) if review_data_path.endswith('.parquet') else pd.read_csv(
            review_data_path)

        # Response templates by tone and sentiment
        self.response_templates = self._initialize_response_templates()

        # Issue keywords for problem identification
        self.issue_keywords = self._initialize_issue_keywords()

    def _initialize_response_templates(self) -> Dict[str, Dict[str, List[str]]]:
        """Initialize response templates by tone and sentiment"""
        return {
            "professional": {
                "positive": [
                    "Thank you for taking the time to share your positive experience with us! We're delighted to hear that {highlight}. Your feedback means a lot to our team, and we look forward to serving you again soon.",
                    "We truly appreciate your wonderful review! It's fantastic to know that {highlight}. We're committed to maintaining this level of excellence and hope to see you again.",
                    "Your positive feedback brightens our day! We're thrilled that {highlight}. Thank you for choosing us, and we can't wait to welcome you back."
                ],
                "negative": [
                    "Thank you for bringing this to our attention. We sincerely apologize that {issue}. This does not reflect our usual standards, and we're taking immediate steps to address this. Please contact us directly at {contact} so we can make this right.",
                    "We're genuinely sorry to hear about your disappointing experience. The issues you've described regarding {issue} are not acceptable, and we're working to improve. We'd appreciate the opportunity to discuss this further and make amends.",
                    "Your feedback is invaluable, and we apologize for falling short of your expectations with {issue}. We're committed to doing better and would welcome the chance to restore your confidence in us."
                ],
                "neutral": [
                    "Thank you for your honest feedback about your experience. We appreciate you taking the time to share your thoughts about {highlight}. We're always looking for ways to improve, and your input helps us do that.",
                    "We value your review and the insights you've provided about {highlight}. Your feedback helps us understand what's working well and where we can enhance our service.",
                    "Thank you for choosing us and for sharing your experience. We've noted your comments about {highlight} and will use this feedback to continue improving."
                ]
            },
            "friendly": {
                "positive": [
                    "Wow, thank you so much for this amazing review! We're over the moon that {highlight}. You've made our whole team smile today! Can't wait to see you again soon!",
                    "This review just made our day! We're so happy that {highlight}. Thanks for being such an awesome customer - you're the best!",
                    "You're absolutely wonderful! Thank you for sharing how much you enjoyed {highlight}. Reviews like yours are why we love what we do!"
                ],
                "negative": [
                    "Oh no! We're really sorry that {issue}. That's definitely not the experience we want for our customers! Please give us a call at {contact} - we'd love to make this up to you and show you the great experience you deserve.",
                    "We're so sorry about what happened with {issue}! This isn't like us at all, and we want to fix it right away. Can you reach out to us directly? We promise we'll make it right!",
                    "Yikes! We really dropped the ball with {issue}. We're genuinely sorry and want to earn back your trust. Please contact us so we can turn this around!"
                ],
                "neutral": [
                    "Thanks for taking the time to review us! We appreciate your thoughts about {highlight}. We're always working to get better, and feedback like yours really helps!",
                    "Thank you for the honest review! It's great to hear your perspective on {highlight}. We'll definitely keep your comments in mind as we continue improving!",
                    "We appreciate you choosing us and sharing your experience! Your feedback about {highlight} is really valuable to us."
                ]
            },
            "formal": {
                "positive": [
                    "We are honored by your positive review and grateful for your patronage. It is most gratifying to learn that {highlight}. We remain committed to providing exceptional service and look forward to serving you again.",
                    "Your commendable review is deeply appreciated. We are pleased that {highlight} met your expectations. We shall continue to uphold these standards of excellence.",
                    "We extend our heartfelt gratitude for your positive feedback. Knowing that {highlight} is truly rewarding for our entire organization."
                ],
                "negative": [
                    "We extend our sincere apologies for the service shortfall you experienced. The matter concerning {issue} is being addressed with the utmost priority. Please contact our management team at {contact} to discuss resolution.",
                    "We deeply regret that our service did not meet your reasonable expectations regarding {issue}. This matter is being investigated thoroughly, and we would welcome the opportunity to rectify this situation.",
                    "Please accept our formal apology for the inconvenience caused by {issue}. We are implementing corrective measures and would appreciate the opportunity to restore your confidence in our establishment."
                ],
                "neutral": [
                    "We acknowledge receipt of your review and appreciate your constructive feedback regarding {highlight}. Your observations will be carefully considered in our ongoing service enhancement efforts.",
                    "Thank you for your thoughtful review. We value your perspective on {highlight} and will incorporate your feedback into our continuous improvement initiatives.",
                    "We appreciate your patronage and the time you have taken to provide feedback about {highlight}. Your input contributes to our commitment to service excellence."
                ]
            }
        }

    def _initialize_issue_keywords(self) -> Dict[str, List[str]]:
        """Initialize keywords for identifying specific issues in reviews"""
        return {
            "service": ["rude", "unfriendly", "slow service", "ignored", "waited", "staff", "server", "employee"],
            "quality": ["poor quality", "bad", "terrible", "awful", "disgusting", "stale", "cold", "overcooked"],
            "cleanliness": ["dirty", "unclean", "messy", "filthy", "gross", "unsanitary", "smell"],
            "value": ["expensive", "overpriced", "not worth", "too much", "costly", "price"],
            "wait_time": ["long wait", "slow", "delayed", "took forever", "waiting", "time"],
            "ambiance": ["noisy", "loud", "uncomfortable", "cramped", "atmosphere", "environment"],
            "management": ["manager", "complaint", "refund", "policy", "refused", "denied"]
        }

    def _analyze_review_sentiment(self, review_text: str) -> str:
        """Analyze review sentiment - simplified sentiment detection"""
        review_lower = review_text.lower()

        positive_words = ["good", "great", "excellent", "amazing", "love", "best", "wonderful", "fantastic", "perfect",
                          "awesome"]
        negative_words = ["bad", "terrible", "awful", "worst", "hate", "horrible", "disappointing", "poor",
                          "disgusting"]

        positive_count = sum(1 for word in positive_words if word in review_lower)
        negative_count = sum(1 for word in negative_words if word in review_lower)

        if negative_count > positive_count:
            return "negative"
        elif positive_count > negative_count:
            return "positive"
        else:
            return "neutral"

    def _identify_specific_issues(self, review_text: str) -> List[str]:
        """Identify specific issues mentioned in the review"""
        review_lower = review_text.lower()
        identified_issues = []

        for issue_type, keywords in self.issue_keywords.items():
            for keyword in keywords:
                if keyword in review_lower:
                    identified_issues.append(issue_type)
                    break  # Only add each issue type once

        return identified_issues

    def _extract_highlights(self, review_text: str, sentiment: str) -> str:
        """Extract key points to highlight in the response"""
        review_lower = review_text.lower()

        if sentiment == "positive":
            # Look for positive aspects mentioned
            highlights = []
            if any(word in review_lower for word in ["food", "meal", "dish", "taste"]):
                highlights.append("you enjoyed our food")
            if any(word in review_lower for word in ["service", "staff", "server", "friendly"]):
                highlights.append("our team provided great service")
            if any(word in review_lower for word in ["atmosphere", "ambiance", "environment"]):
                highlights.append("you loved the atmosphere")
            if any(word in review_lower for word in ["clean", "fresh", "quality"]):
                highlights.append("you appreciated our quality")

            return ", ".join(highlights) if highlights else "you had a positive experience"

        elif sentiment == "negative":
            # Identify main issues
            issues = self._identify_specific_issues(review_text)
            if issues:
                issue_descriptions = {
                    "service": "your service experience wasn't up to our standards",
                    "quality": "the quality didn't meet your expectations",
                    "cleanliness": "cleanliness was an issue",
                    "value": "you felt the value wasn't there",
                    "wait_time": "you experienced longer wait times than expected",
                    "ambiance": "the atmosphere wasn't comfortable",
                    "management": "there was a management issue"
                }
                return issue_descriptions.get(issues[0], "your experience wasn't satisfactory")
            else:
                return "your experience wasn't satisfactory"

        else:  # neutral
            # General highlights
            if "food" in review_lower:
                return "your thoughts on our food"
            elif "service" in review_lower:
                return "your feedback about our service"
            else:
                return "your experience with us"

    def _get_business_contact_info(self, business_id: str) -> str:
        """Get business contact information for responses"""
        # In a real implementation, this would get actual contact info
        # For now, return placeholder
        return "our customer service team"

    def _generate_response(self, review_text: str, sentiment: str, tone: str, business_id: str) -> str:
        """Generate the actual response based on sentiment and tone"""
        if tone not in self.response_templates:
            tone = "professional"  # Default fallback

        if sentiment not in self.response_templates[tone]:
            sentiment = "neutral"  # Default fallback

        # Get appropriate template
        templates = self.response_templates[tone][sentiment]
        template = templates[0]  # Use first template (could be randomized)

        # Extract highlights/issues
        highlight_text = self._extract_highlights(review_text, sentiment)

        # Get contact info if needed
        contact_info = self._get_business_contact_info(business_id)

        # Format the response step by step
        response = template

        # Replace highlight/issue placeholders
        if "{highlight}" in response:
            response = response.replace("{highlight}", highlight_text)
        elif "{issue}" in response:
            response = response.replace("{issue}", highlight_text)

        # Replace contact placeholder if it exists
        if "{contact}" in response:
            response = response.replace("{contact}", contact_info)

        return response

    def _generate_key_points(self, review_text: str, sentiment: str) -> List[str]:
        """Generate key points that should be addressed in the response"""
        points = []

        if sentiment == "positive":
            points.append("Thank the customer for positive feedback")
            points.append("Acknowledge specific positives mentioned")
            points.append("Invite them to return")

        elif sentiment == "negative":
            issues = self._identify_specific_issues(review_text)
            points.append("Apologize for the poor experience")

            for issue in issues:
                if issue == "service":
                    points.append("Address service quality concerns")
                elif issue == "quality":
                    points.append("Acknowledge quality issues")
                elif issue == "wait_time":
                    points.append("Address timing concerns")
                # Add more specific issue handling

            points.append("Offer to resolve the issue directly")
            points.append("Provide contact information")

        else:  # neutral
            points.append("Thank customer for honest feedback")
            points.append("Acknowledge their specific comments")
            points.append("Show commitment to improvement")

        return points

    def _assess_escalation_flags(self, review_text: str, sentiment: str) -> List[str]:
        """Identify if the review requires escalation"""
        flags = []
        review_lower = review_text.lower()

        # High priority flags
        if any(word in review_lower for word in ["lawyer", "legal", "sue", "health department"]):
            flags.append("legal_threat")

        if any(word in review_lower for word in ["sick", "food poisoning", "illness", "hospital"]):
            flags.append("health_concern")

        if any(word in review_lower for word in ["discrimination", "racist", "harassment"]):
            flags.append("discrimination_claim")

        # Medium priority flags
        if sentiment == "negative" and any(word in review_lower for word in ["never", "again", "worst", "horrible"]):
            flags.append("high_negative_sentiment")

        if "refund" in review_lower or "money back" in review_lower:
            flags.append("refund_request")

        return flags

    def __call__(self, business_id: str, review_text: str, response_tone: str = "professional") -> Dict[str, Any]:
        """
        Main callable method - follows same pattern as other tools

        Args:
            business_id: Target business identifier
            review_text: The review content to respond to
            response_tone: Tone for the response ("professional", "friendly", "formal")
        """

        # Validate inputs
        if not review_text.strip():
            return {
                "business_id": business_id,
                "error": "Review text cannot be empty",
                "response": "",
                "key_points": [],
                "escalation_flags": []
            }

        # Analyze the review
        sentiment = self._analyze_review_sentiment(review_text)
        specific_issues = self._identify_specific_issues(review_text)

        # Generate response
        generated_response = self._generate_response(review_text, sentiment, response_tone, business_id)

        # Generate key points to address
        key_points = self._generate_key_points(review_text, sentiment)

        # Check for escalation needs
        escalation_flags = self._assess_escalation_flags(review_text, sentiment)

        # Return structured result
        return {
            "business_id": business_id,
            "review_analysis": {
                "sentiment": sentiment,
                "specific_issues": specific_issues,
                "review_length": len(review_text.split()),
                "urgency_level": "high" if escalation_flags else "medium" if sentiment == "negative" else "low"
            },
            "generated_response": generated_response,
            "response_tone": response_tone,
            "key_points_addressed": key_points
        }