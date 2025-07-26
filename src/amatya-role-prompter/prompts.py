"""
Role-specific prompt construction for advice generation.

This module handles the construction of prompts tailored to different user roles,
incorporating code snippets and context for optimal LLM responses.
"""

import logging
from typing import List, Dict
from common.models import Snippet

logger = logging.getLogger(__name__)


class PromptConstructor:
    """Constructs role-specific prompts for advice generation."""
    
    def __init__(self):
        """Initialize the prompt constructor."""
        self.role_templates = self._load_role_templates()
        logger.info("PromptConstructor initialized")
    
    def construct_prompt(self, role: str, chunks: List[Snippet]) -> str:
        """
        Construct a role-specific prompt for advice generation.
        
        Args:
            role: User role (e.g., 'backend_developer', 'security_engineer')
            chunks: List of code snippets for context
            
        Returns:
            str: Constructed prompt for the LLM
        """
        # Build context from chunks
        context = self._build_context(chunks)
        
        # Get role-specific template
        template = self._get_role_template(role)
        
        # Construct the final prompt
        prompt = template.format(
            role=role.replace('_', ' ').title(),
            context=context,
            num_files=len(chunks)
        )
        
        logger.info(f"Constructed prompt for role '{role}' with {len(chunks)} chunks")
        return prompt
    
    def _build_context(self, chunks: List[Snippet]) -> str:
        """
        Build context string from code snippets.
        
        Args:
            chunks: List of code snippets
            
        Returns:
            str: Formatted context string
        """
        if not chunks:
            return "No code snippets provided."
        
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            context_parts.append(
                f"**File {i}: {chunk.file_path}**\n```\n{chunk.content}\n```"
            )
        
        return "\n\n".join(context_parts)
    
    def _get_role_template(self, role: str) -> str:
        """
        Get the appropriate template for a role.
        
        Args:
            role: User role
            
        Returns:
            str: Template string for the role
        """
        # Normalize role name
        normalized_role = role.lower().replace(' ', '_')
        
        # Return specific template if available, otherwise use default
        return self.role_templates.get(normalized_role, self.role_templates['default'])
    
    def _load_role_templates(self) -> Dict[str, str]:
        """Load role-specific prompt templates."""
        return {
            'default': """You are an experienced onboarding assistant helping a new {role} understand and contribute to this project.

You have access to the following code excerpts and documentation:

{context}

Based on this information, provide a comprehensive onboarding guide for the {role}. Your response should:

1. **Overview**: Briefly explain what this project does and the {role}'s role in it
2. **Getting Started**: Step-by-step guidance for setting up and understanding the codebase
3. **Key Areas**: Highlight the most important files and concepts for this role
4. **Best Practices**: Share relevant coding standards and patterns used in this project
5. **Next Steps**: Suggest specific tasks or areas to explore further

Format your response in clear markdown with:
- Numbered lists for step-by-step instructions
- Code references using backticks
- File path references for each recommendation
- At least 3 specific file references from the provided context

Be concise but thorough, focusing on actionable guidance that will help the {role} become productive quickly.""",

            'backend_developer': """You are a senior backend developer helping onboard a new backend developer to this project.

You have access to the following code excerpts and documentation:

{context}

As an experienced backend developer, provide a comprehensive onboarding guide focusing on:

1. **Architecture Overview**: Explain the backend architecture, API design, and data flow
2. **Development Setup**: Guide through environment setup, dependencies, and local development
3. **Core Backend Components**: Highlight key services, models, and business logic
4. **API Patterns**: Explain REST/GraphQL patterns, authentication, and error handling
5. **Database & Storage**: Cover data models, migrations, and storage patterns
6. **Testing Strategy**: Explain unit tests, integration tests, and testing patterns
7. **Deployment & DevOps**: Cover CI/CD, containerization, and deployment processes

Include specific file references and code examples. Focus on backend-specific concerns like performance, scalability, and maintainability.""",

            'frontend_developer': """You are a senior frontend developer helping onboard a new frontend developer to this project.

You have access to the following code excerpts and documentation:

{context}

As an experienced frontend developer, provide a comprehensive onboarding guide focusing on:

1. **UI/UX Architecture**: Explain the frontend architecture, component structure, and design system
2. **Development Environment**: Guide through setup, build tools, and development workflow
3. **Component Library**: Highlight reusable components, styling patterns, and UI conventions
4. **State Management**: Explain data flow, state management patterns, and API integration
5. **Routing & Navigation**: Cover page structure, routing patterns, and user flows
6. **Performance**: Discuss optimization strategies, bundling, and loading patterns
7. **Testing**: Explain component testing, e2e testing, and testing utilities

Include specific file references and focus on frontend-specific concerns like user experience, accessibility, and browser compatibility.""",

            'security_engineer': """You are a senior security engineer helping onboard a new security engineer to this project.

You have access to the following code excerpts and documentation:

{context}

As an experienced security engineer, provide a comprehensive onboarding guide focusing on:

1. **Security Architecture**: Explain authentication, authorization, and security boundaries
2. **Threat Model**: Identify potential security risks and mitigation strategies
3. **Authentication & Authorization**: Cover user management, tokens, and access controls
4. **Data Protection**: Explain encryption, data handling, and privacy considerations
5. **Input Validation**: Highlight validation patterns and injection prevention
6. **Security Testing**: Cover security testing tools, penetration testing, and vulnerability scanning
7. **Compliance**: Discuss relevant standards, regulations, and audit requirements

Include specific file references and focus on security-specific concerns like threat vectors, secure coding practices, and incident response.""",

            'devops_engineer': """You are a senior DevOps engineer helping onboard a new DevOps engineer to this project.

You have access to the following code excerpts and documentation:

{context}

As an experienced DevOps engineer, provide a comprehensive onboarding guide focusing on:

1. **Infrastructure Overview**: Explain the deployment architecture, cloud services, and infrastructure as code
2. **CI/CD Pipeline**: Detail the build, test, and deployment processes
3. **Containerization**: Cover Docker, Kubernetes, and container orchestration
4. **Monitoring & Logging**: Explain observability, metrics, alerts, and log aggregation
5. **Security & Compliance**: Cover infrastructure security, secrets management, and compliance
6. **Scaling & Performance**: Discuss auto-scaling, load balancing, and performance optimization
7. **Disaster Recovery**: Explain backup strategies, failover, and incident response

Include specific file references and focus on operational concerns like reliability, scalability, and maintainability.""",

            'data_scientist': """You are a senior data scientist helping onboard a new data scientist to this project.

You have access to the following code excerpts and documentation:

{context}

As an experienced data scientist, provide a comprehensive onboarding guide focusing on:

1. **Data Architecture**: Explain data sources, pipelines, and storage systems
2. **ML/AI Components**: Highlight machine learning models, training pipelines, and inference systems
3. **Data Processing**: Cover ETL processes, data cleaning, and feature engineering
4. **Model Development**: Explain model training, validation, and experimentation workflows
5. **Data Analysis**: Discuss analytics tools, visualization, and reporting systems
6. **Model Deployment**: Cover model serving, monitoring, and A/B testing
7. **Data Governance**: Explain data quality, privacy, and compliance requirements

Include specific file references and focus on data-specific concerns like model performance, data quality, and reproducibility."""
        }
    
    def get_available_roles(self) -> List[str]:
        """Get list of available role templates."""
        return list(self.role_templates.keys())
