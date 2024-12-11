from git import Repo, InvalidGitRepositoryError

repo = Repo("/Users/ibk5106/Desktop/IST_courses/TA/wge")

print(repo.is_dirty())