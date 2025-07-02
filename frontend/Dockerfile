FROM node:20-alpine as builder
WORKDIR /app

# Install dependencies
COPY package*.json ./
RUN npm install

# Copy source
COPY . .

# Build static files
ARG VITE_API_BASE=http://backend:8000/api
ENV VITE_API_BASE=$VITE_API_BASE
RUN npm run build

# -- Runtime stage --
FROM node:20-alpine
WORKDIR /app
RUN npm install -g serve
COPY --from=builder /app/dist ./dist
EXPOSE 3000
CMD ["serve", "-s", "dist", "-l", "3000"] 