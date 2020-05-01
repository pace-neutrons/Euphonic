def call() {
    withCredentials([string(credentialsId: 'Euphonic_GitHub_API_Token',
            variable: 'api_token')]) {
        String GITHUB_API_BRANCH_URL = "https://api.github.com/repos/pace-neutrons/Euphonic/git/ref/heads/${env.JOB_BASE_NAME}"
        def branch_url_response = GITHUB_API_BRANCH_URL.toURL().getText(requestProperties: ['Authorization': "token ${api_token}"])
        def commit_url = new JsonSlurper().parseText(branch_url_response).object.url
        def commit_url_response = commit_url.toURL().getText(requestProperties: ['Authorization': "token ${api_token}"])
        return new JsonSlurper().parseText(commit_url_response).author.email
    }
}