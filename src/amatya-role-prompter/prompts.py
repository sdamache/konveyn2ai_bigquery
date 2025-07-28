"""
Role-specific prompt construction for advice generation.

This module handles the construction of prompts tailored to different user roles,
incorporating code snippets and context for optimal LLM responses.
"""

import logging

from common.models import Snippet

logger = logging.getLogger(__name__)


class PromptConstructor:
    """Constructs role-specific prompts for advice generation."""

    def __init__(self):
        """Initialize the prompt constructor."""
        self.role_templates = self._load_role_templates()
        logger.info("PromptConstructor initialized")

    def construct_prompt(self, role: str, question: str, chunks: list[Snippet]) -> str:
        """
        Construct a role-specific prompt for advice generation.

        Args:
            role: User role (e.g., 'backend_developer', 'security_engineer')
            question: User's specific question to be answered
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
            role=role.replace("_", " ").title(),
            question=question,
            context=context,
            num_files=len(chunks),
        )

        logger.info(f"Constructed prompt for role '{role}' with {len(chunks)} chunks")
        return prompt

    def _build_context(self, chunks: list[Snippet]) -> str:
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
        normalized_role = role.lower().replace(" ", "_")

        # Return specific template if available, otherwise use default
        return self.role_templates.get(normalized_role, self.role_templates["default"])

    def _load_role_templates(self) -> dict[str, str]:
        """Load role-specific prompt templates."""
        return {
            "default": """You are an experienced {role} helping answer a specific technical question.

**User's Question**: {question}

You have access to the following relevant code excerpts and documentation:

{context}

Based on this information, provide a comprehensive answer to the user's question from a {role} perspective. Your response should:

1. **Direct Answer**: Address the specific question asked
2. **Implementation Details**: Provide concrete steps and code examples
3. **Best Practices**: Share relevant patterns and standards for this approach
4. **Code References**: Reference the provided code snippets where relevant
5. **Additional Considerations**: Mention important related concepts or potential issues

Format your response in clear markdown with:
- Specific code examples and snippets
- File path references from the provided context
- Step-by-step implementation guidance
- Security, performance, or other relevant considerations for a {role}

Focus on providing actionable, specific guidance that directly answers the question asked.""",
            "backend_developer": """You are a senior backend developer helping answer a specific technical question.

**User's Question**: {question}

You have access to the following relevant code excerpts and documentation:

{context}

As an experienced backend developer, provide a comprehensive answer focusing on:

1. **Architecture Overview**: Explain the backend architecture, API design, and data flow
2. **Development Setup**: Guide through environment setup, dependencies, and local development
3. **Core Backend Components**: Highlight key services, models, and business logic
4. **API Patterns**: Explain REST/GraphQL patterns, authentication, and error handling
5. **Database & Storage**: Cover data models, migrations, and storage patterns
6. **Testing Strategy**: Explain unit tests, integration tests, and testing patterns
7. **Deployment & DevOps**: Cover CI/CD, containerization, and deployment processes

Include specific file references and code examples. Focus on backend-specific concerns like performance, scalability, and maintainability.""",
            "frontend_developer": """You are a senior frontend developer helping answer a specific technical question.

**User's Question**: {question}

You have access to the following relevant code excerpts and documentation:

{context}

As an experienced frontend developer, provide a comprehensive answer focusing on:

1. **Direct Answer**: Address the specific question asked from a frontend perspective
2. **Implementation Details**: Provide concrete steps and code examples
3. **UI/UX Considerations**: Explain relevant design patterns and user experience aspects
4. **Technical Implementation**: Cover component structure, state management, and API integration
5. **Best Practices**: Share frontend-specific patterns and standards
6. **Performance & Accessibility**: Discuss optimization and accessibility considerations

Include specific file references and code examples. Focus on providing actionable guidance that directly answers the question asked.""",
            "security_engineer": """You are a senior security engineer helping answer a specific technical question.

**User's Question**: {question}

You have access to the following relevant code excerpts and documentation:

{context}

As an experienced security engineer, provide a comprehensive answer focusing on:

1. **Direct Answer**: Address the specific security question asked
2. **Security Implementation**: Provide concrete security measures and code examples
3. **Threat Analysis**: Identify relevant security risks and mitigation strategies
4. **Best Practices**: Share security-specific patterns and standards
5. **Compliance Considerations**: Discuss relevant security standards and requirements
6. **Testing & Validation**: Explain security testing approaches for the implementation

Include specific file references and code examples. Focus on providing actionable security guidance that directly answers the question asked.""",
            "devops_engineer": """You are a senior DevOps engineer helping answer a specific technical question.

**User's Question**: {question}

You have access to the following relevant code excerpts and documentation:

{context}

As an experienced DevOps engineer, provide a comprehensive answer focusing on:

1. **Direct Answer**: Address the specific DevOps question asked
2. **Implementation Details**: Provide concrete infrastructure and deployment steps
3. **Best Practices**: Share DevOps-specific patterns and operational standards
4. **Infrastructure Considerations**: Explain relevant cloud services and architecture
5. **CI/CD Integration**: Discuss build, test, and deployment pipeline aspects
6. **Monitoring & Security**: Cover observability and security considerations

Include specific file references and code examples. Focus on providing actionable DevOps guidance that directly answers the question asked.""",
            "data_scientist": """You are a senior data scientist helping answer a specific technical question.

**User's Question**: {question}

You have access to the following relevant code excerpts and documentation:

{context}

As an experienced data scientist, provide a comprehensive answer focusing on:

1. **Data Architecture**: Explain data sources, pipelines, and storage systems
2. **ML/AI Components**: Highlight machine learning models, training pipelines, and inference systems
3. **Data Processing**: Cover ETL processes, data cleaning, and feature engineering
4. **Model Development**: Explain model training, validation, and experimentation workflows
5. **Data Analysis**: Discuss analytics tools, visualization, and reporting systems
6. **Model Deployment**: Cover model serving, monitoring, and A/B testing
7. **Data Governance**: Explain data quality, privacy, and compliance requirements

Include specific file references and focus on data-specific concerns like model performance, data quality, and reproducibility.""",
            "product_manager": """You are a senior product manager helping onboard a new product manager to this project.

You have access to the following code excerpts and documentation:

{context}

As an experienced product manager, provide a comprehensive onboarding guide focusing on:

1. **Product Overview**: Explain the product vision, user personas, and key features
2. **Technical Architecture**: Understand the system design and technical constraints
3. **User Journey**: Map out user flows and interaction patterns
4. **Feature Prioritization**: Explain how features are planned and prioritized
5. **Metrics & Analytics**: Cover key performance indicators and measurement strategies
6. **Stakeholder Management**: Identify key stakeholders and communication patterns
7. **Release Process**: Understand deployment cycles and feature rollout strategies

Include specific file references and focus on product-specific concerns like user experience, business impact, and technical feasibility.""",
            "qa_engineer": """You are a senior QA engineer helping answer a specific technical question.

**User's Question**: {question}

You have access to the following relevant code excerpts and documentation:

{context}

As an experienced QA engineer, provide a comprehensive answer focusing on:

1. **Direct Answer**: Address the specific testing question asked
2. **Implementation Details**: Provide concrete testing steps and code examples
3. **Best Practices**: Share QA-specific patterns and testing standards
4. **Test Strategy**: Explain relevant testing approaches and methodologies
5. **Quality Considerations**: Discuss coverage, reliability, and validation aspects
6. **Tools & Frameworks**: Recommend appropriate testing tools and frameworks

Include specific file references and code examples. Focus on providing actionable QA guidance that directly answers the question asked.""",
            "technical_writer": """You are a senior technical writer helping answer a specific technical question.

**User's Question**: {question}

You have access to the following relevant code excerpts and documentation:

{context}

As an experienced technical writer, provide a comprehensive answer focusing on:

1. **Direct Answer**: Address the specific documentation question asked
2. **Implementation Details**: Provide concrete documentation steps and examples
3. **Best Practices**: Share technical writing patterns and standards
4. **Content Strategy**: Explain relevant documentation approaches and structure
5. **Tools & Formats**: Recommend appropriate documentation tools and formats
6. **Audience Considerations**: Discuss clarity, accuracy, and maintainability aspects

Include specific file references and code examples. Focus on providing actionable documentation guidance that directly answers the question asked.""",
        }

    def get_available_roles(self) -> list[str]:
        """Get list of available role templates."""
        return list(self.role_templates.keys())

    def get_context_aware_prompt(
        self, role: str, question: str, chunks: list[Snippet]
    ) -> str:
        """
        Get a context-aware prompt that adapts based on the code content.

        Args:
            role: User role
            question: User's specific question
            chunks: List of code snippets

        Returns:
            str: Context-aware prompt
        """
        # Analyze the code content to determine project characteristics
        project_context = self._analyze_project_context(chunks)

        # Get base template
        base_prompt = self.construct_prompt(role, question, chunks)

        # Add context-specific guidance
        if project_context["has_api"]:
            base_prompt += "\n\n**API Integration Notes**: This project includes API endpoints. Pay special attention to request/response handling, authentication, and error management."

        if project_context["has_database"]:
            base_prompt += "\n\n**Database Integration Notes**: This project includes database operations. Consider data modeling, migrations, and query optimization."

        if project_context["has_auth"]:
            base_prompt += "\n\n**Security Notes**: This project includes authentication/authorization. Focus on security best practices and access control patterns."

        if project_context["has_tests"]:
            base_prompt += "\n\n**Testing Notes**: This project includes test files. Review the testing strategy and ensure new code follows established patterns."

        return base_prompt

    def _analyze_project_context(self, chunks: list[Snippet]) -> dict[str, bool]:
        """
        Analyze code chunks to understand project characteristics.

        Args:
            chunks: List of code snippets

        Returns:
            dict: Project characteristics
        """
        context = {
            "has_api": False,
            "has_database": False,
            "has_auth": False,
            "has_tests": False,
            "has_frontend": False,
            "has_docker": False,
        }

        for chunk in chunks:
            content_lower = chunk.content.lower()
            file_path_lower = chunk.file_path.lower()

            # Check for API patterns
            if any(
                keyword in content_lower
                for keyword in ["@app.", "fastapi", "flask", "router", "endpoint"]
            ):
                context["has_api"] = True

            # Check for database patterns
            if any(
                keyword in content_lower
                for keyword in [
                    "database",
                    "db.",
                    "session",
                    "query",
                    "model",
                    "sqlalchemy",
                ]
            ):
                context["has_database"] = True

            # Check for authentication patterns
            if any(
                keyword in content_lower
                for keyword in ["auth", "login", "token", "jwt", "password", "security"]
            ):
                context["has_auth"] = True

            # Check for test files
            if any(
                keyword in file_path_lower
                for keyword in ["test_", "_test", "tests/", "spec_"]
            ):
                context["has_tests"] = True

            # Check for frontend patterns
            if any(
                keyword in content_lower
                for keyword in ["react", "vue", "angular", "component", "jsx", "tsx"]
            ):
                context["has_frontend"] = True

            # Check for Docker
            if "dockerfile" in file_path_lower or "docker" in content_lower:
                context["has_docker"] = True

        return context
