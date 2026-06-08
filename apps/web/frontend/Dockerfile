FROM node:20-alpine

WORKDIR /app

COPY apps/web/frontend/package*.json ./
RUN npm ci

COPY apps/web/frontend ./

CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0"]
